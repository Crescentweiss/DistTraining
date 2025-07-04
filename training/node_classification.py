import argparse
import socket
import time
import logging
import os

import dgl
import dgl.dataloading
import dgl.distributed
import dgl.nn.pytorch as dglnn
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

import sys
import os
sys.path.append('/N/slate/yuzih/DGL/DistTraining')
from model.graphmodel import *
from profiler.communication import *

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted labels.
    labels : torch.Tensor
        Ground-truth labels.

    Returns
    -------
    float
        Accuracy.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, inputs, labels, val_nid, test_nid, batch_size, device):
    """
    Evaluate the model on the validation and test set.

    Parameters
    ----------
    model : DistSAGE
        The model to be evaluated.
    g : DistGraph
        The entire graph.
    inputs : DistTensor
        The feature data of all the nodes.
    labels : DistTensor
        The labels of all the nodes.
    val_nid : torch.Tensor
        The node IDs for validation.
    test_nid : torch.Tensor
        The node IDs for test.
    batch_size : int
        Batch size for evaluation.
    device : torch.Device
        The target device to evaluate on.

    Returns
    -------
    float
        Validation accuracy.
    float
        Test accuracy.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(
        pred[test_nid], labels[test_nid]
    )


def run(args, device, data):
    """
    Train and evaluate DistSAGE.

    Parameters
    ----------
    args : argparse.Args
        Arguments for train and evaluate.
    device : torch.Device
        Target device for train and evaluate.
    data : Packed Data
        Packed data includes train/val/test IDs, feature dimension,
        number of classes, graph.
    """
    train_nid, val_nid, test_nid, in_feats, n_classes, g = data
    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )
    dataloader = dgl.distributed.DistNodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    model = DistSAGE(
        in_feats,
        args.num_hidden,
        n_classes,
        args.num_layers,
        F.relu,
        args.dropout,
    )
    model = model.to(device)
    if args.num_gpus == 0:
        model = th.nn.parallel.DistributedDataParallel(model)
    else:
        model = th.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop.
    iter_tput = []
    epoch = 0
    epoch_time = []
    test_acc = 0.0
    for _ in range(args.num_epochs):
        epoch += 1
        tic = time.time()
        # Various time statistics.
        sample_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0
        start = time.time()
        step_time = []

        # 新增：通信时间统计
        total_local_time = 0
        total_remote_time = 0
        total_local_nodes = 0
        total_remote_nodes = 0

        with model.join():
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                tic_step = time.time()
                sample_time += tic_step - start
                # # Slice feature and label.
                # batch_inputs = g.ndata["features"][input_nodes]
                # batch_labels = g.ndata["labels"][seeds].long()

                # 分析当前batch的通信需求
                comm_analysis = analyze_cross_node_comm(g, input_nodes, blocks, trainers_per_node=2)

                # 测量特征访问时间
                feature_times = time_feature_access(g, 
                                                comm_analysis['local_node_ids'], 
                                                comm_analysis['remote_node_ids'])
                
                # 累积时间统计
                total_local_time += feature_times['local_time']
                total_remote_time += feature_times['remote_time']
                total_local_nodes += feature_times['local_count']
                total_remote_nodes += feature_times['remote_count']
                # 注意：这里不再重复获取特征，因为time_feature_access已经获取过了
                # 直接使用time_feature_access的结果或重新获取
                batch_inputs = g.ndata["features"][input_nodes]
                batch_labels = g.ndata["labels"][seeds].long()

                # 打印通信分析结果
                if step % args.log_every == 0:
                    import socket
                    current_host = socket.gethostname()
                    current_rank = g.rank()
                    device_info = f"GPU:{device}" if th.cuda.is_available() else "CPU"
                    
                    # 计算平均时间
                    avg_local_time = (feature_times['local_time'] / feature_times['local_count'] * 1000 
                                    if feature_times['local_count'] > 0 else 0)
                    avg_remote_time = (feature_times['remote_time'] / feature_times['remote_count'] * 1000 
                                    if feature_times['remote_count'] > 0 else 0)
                    
                    print(f"[{current_host}|Rank:{current_rank}|{device_info}] Batch {step}: "
                        f"Remote: {comm_analysis['remote_nodes']}/{comm_analysis['total_nodes']} "
                        f"({comm_analysis['remote_ratio']*100:.1f}%), "
                        f"Comm: {comm_analysis['estimated_comm_bytes']/1024:.1f} KB")
                    
                    print(f"[{current_host}|Rank:{current_rank}] Feature Time: "
                        f"Local: {feature_times['local_time']*1000:.2f}ms "
                        f"({feature_times['local_count']} nodes, {avg_local_time:.3f}ms/node), "
                        f"Remote: {feature_times['remote_time']*1000:.2f}ms "
                        f"({feature_times['remote_count']} nodes, {avg_remote_time:.3f}ms/node)")
                    
                    print(f"[{current_host}|Rank:{current_rank}] Partitions: {comm_analysis['partition_distribution']}")
                
                num_seeds += len(blocks[-1].dstdata[dgl.NID])
                num_inputs += len(blocks[0].srcdata[dgl.NID])
                # # Move to target device.
                # blocks = [block.to(device) for block in blocks]
                # batch_inputs = batch_inputs.to(device)
                # batch_labels = batch_labels.to(device)
                # 统计CPU到GPU的数据传输时间
                device_transfer_start = time.time()
                blocks = [block.to(device) for block in blocks]
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                device_transfer_time = time.time() - device_transfer_start
                # 累积设备传输时间统计
                if 'total_device_transfer_time' not in locals():
                    total_device_transfer_time = 0
                    device_transfer_count = 0
                total_device_transfer_time += device_transfer_time
                device_transfer_count += 1
                # 定期打印设备传输时间
                if step % args.log_every == 0:
                    avg_transfer_time = total_device_transfer_time / device_transfer_count * 1000
                    print(f"[GPU Transfer] Step {step}: Current {device_transfer_time*1000:.2f}ms, "
                          f"Average {avg_transfer_time:.2f}ms per batch")

                # Compute loss and prediction.
                start = time.time()
                
                ############# 手动profiler结束 #############
                # 创建profiler（只在特定step启用以避免性能开销）
                if step % args.log_every == 0 and step < 100:  # 只在前100个step的log步骤启用
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=True,
                        with_flops=True,
                        with_modules=True
                    ) as prof:
                        with record_function("forward_pass"):
                            batch_pred = model(blocks, batch_inputs)
                        
                        with record_function("loss_computation"):
                            loss = loss_fcn(batch_pred, batch_labels)
                        
                        forward_end = time.time()
                        
                        with record_function("backward_pass"):
                            optimizer.zero_grad()
                            loss.backward()
                        
                        compute_end = time.time()
                        
                        with record_function("parameter_update"):
                            optimizer.step()
                    # 保存profiler结果
                    trace_dir = "/N/slate/yuzih/DGL/DistTraining/profiler_traces/"+ args.graph_name
                    os.makedirs(trace_dir, exist_ok=True)
                    # 保存profiler结果到指定目录
                    trace_file = os.path.join(trace_dir, f"trace_rank_{g.rank()}_step_{step}.json")
                    prof.export_chrome_trace(trace_file)
                    print(f"Trace file saved to: {trace_file}")
                    # 打印关键操作的统计信息
                    print(f"\n=== Profiler Results for Rank {g.rank()}, Step {step} ===")
                    # 按函数分组统计
                    print("Top operations by CPU time:")
                    print(prof.key_averages(group_by_input_shape=True).table(
                        sort_by="cpu_time_total", row_limit=10))
                    if th.cuda.is_available():
                        print("\nTop operations by CUDA time:")
                        print(prof.key_averages(group_by_input_shape=True).table(
                            sort_by="cuda_time_total", row_limit=10))
                    # 查找通信相关的操作
                    events = prof.key_averages()
                    comm_events = []
                    for event in events:
                        if any(keyword in event.key.lower() for keyword in 
                              ['DistributedDataParallel.forward', 'backward_pass', 'gloo', 'nccl']):
                            comm_events.append(event)
                    if comm_events:
                        print(f"\n=== Communication Operations ===")
                        for event in comm_events:
                            print(f"{event.key}: CPU {event.cpu_time_total/1000:.2f}ms, "
                                  f"CUDA {event.cuda_time_total/1000:.2f}ms, "
                                  f"Count: {event.count}")
                    print("=" * 60)
                else:
                    # 正常执行，不启用profiler
                    batch_pred = model(blocks, batch_inputs)
                    loss = loss_fcn(batch_pred, batch_labels)
                    forward_end = time.time()
                    optimizer.zero_grad()
                    loss.backward()
                    compute_end = time.time()
                ############# 手动profiler结束 #############

                # batch_pred = model(blocks, batch_inputs)
                # loss = loss_fcn(batch_pred, batch_labels)
                # forward_end = time.time()
                # optimizer.zero_grad()
                # loss.backward()
                # compute_end = time.time()
                # 统计各个阶段的时间
                forward_time += forward_end - start
                backward_time += compute_end - forward_end

                optimizer.step()
                update_time += time.time() - compute_end

                step_t = time.time() - tic_step
                step_time.append(step_t)
                iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
                if (step + 1) % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    gpu_mem_alloc = (
                        th.cuda.max_memory_allocated() / 1000000
                        if th.cuda.is_available()
                        else 0
                    )
                    sample_speed = np.mean(iter_tput[-args.log_every :])
                    mean_step_time = np.mean(step_time[-args.log_every :])
                    print(
                        f"Part {g.rank()} | Epoch {epoch:05d} | Step {step:05d}"
                        f" | Loss {loss.item():.4f} | Train Acc {acc.item():.4f}"
                        f" | Speed (samples/sec) {sample_speed:.4f}"
                        f" | GPU {gpu_mem_alloc:.1f} MB | "
                        f"Mean step time {mean_step_time:.3f} s"
                    )
                start = time.time()

        toc = time.time()

        avg_local_per_node = (total_local_time / total_local_nodes * 1000 
                            if total_local_nodes > 0 else 0)
        avg_remote_per_node = (total_remote_time / total_remote_nodes * 1000 
                            if total_remote_nodes > 0 else 0)
        
        print(f"Part {g.rank()}, Epoch {epoch} Feature Access Summary:")
        print(f"  Local: {total_local_time:.4f}s ({total_local_nodes} nodes, {avg_local_per_node:.3f}ms/node)")
        print(f"  Remote: {total_remote_time:.4f}s ({total_remote_nodes} nodes, {avg_remote_per_node:.3f}ms/node)")
        print(f"  Device Transfer: {total_device_transfer_time:.4f}s ({device_transfer_count} batches, "
               f"{total_device_transfer_time/device_transfer_count*1000:.2f}ms/batch)")
        # print(f"  Local vs Remote speed ratio: {avg_remote_per_node/avg_local_per_node:.2f}x" 
        #     if avg_local_per_node > 0 else "")

        # print(
        #     f"Part {g.rank()}, Epoch Time(s): {toc - tic:.4f}, "
        #     f"sample+data_copy: {sample_time:.4f}, forward: {forward_time:.4f},"
        #     f" backward: {backward_time:.4f}, update: {update_time:.4f}, "
        #     f"#seeds: {num_seeds}, #inputs: {num_inputs}"
        # )
        epoch_time.append(toc - tic)

        if epoch % args.eval_every == 0 or epoch == args.num_epochs:
            start = time.time()
            val_acc, test_acc = evaluate(
                model.module,
                g,
                g.ndata["features"],
                g.ndata["labels"],
                val_nid,
                test_nid,
                args.batch_size_eval,
                device,
            )
            print(
                f"Part {g.rank()}, Val Acc {val_acc:.4f}, "
                f"Test Acc {test_acc:.4f}, time: {time.time() - start:.4f}"
            )

    return np.mean(epoch_time[-int(args.num_epochs * 0.8) :]), test_acc


def main(args):
    """
    Main function.
    """
    host_name = socket.gethostname()
    print(f"{host_name}: Initializing DistDGL.")
    dgl.distributed.initialize(args.ip_config, use_graphbolt=args.use_graphbolt)
    print(f"{host_name}: Initializing PyTorch process group.")
    th.distributed.init_process_group(backend=args.backend)
    print(f"{host_name}: Initializing DistGraph.")
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    print(f"Rank of {host_name}: {g.rank()}")

    # Split train/val/test IDs for each trainer.
    pb = g.get_partition_book()
    if "trainer_id" in g.ndata:
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata["trainer_id"],
        )
    else:
        train_nid = dgl.distributed.node_split(
            g.ndata["train_mask"], pb, force_even=True
        )
        val_nid = dgl.distributed.node_split(
            g.ndata["val_mask"], pb, force_even=True
        )
        test_nid = dgl.distributed.node_split(
            g.ndata["test_mask"], pb, force_even=True
        )
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    num_train_local = len(np.intersect1d(train_nid.numpy(), local_nid))
    num_val_local = len(np.intersect1d(val_nid.numpy(), local_nid))
    num_test_local = len(np.intersect1d(test_nid.numpy(), local_nid))
    print(
        f"part {g.rank()}, train: {len(train_nid)} (local: {num_train_local}), "
        f"val: {len(val_nid)} (local: {num_val_local}), "
        f"test: {len(test_nid)} (local: {num_test_local})"
    )
    del local_nid
    if args.num_gpus == 0:
        device = th.device("cpu")
    else:
        dev_id = g.rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
    n_classes = args.n_classes
    if n_classes == 0:
        labels = g.ndata["labels"][np.arange(g.num_nodes())]
        n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
        del labels
    print(f"Number of classes: {n_classes}")

    # Pack data.
    in_feats = g.ndata["features"].shape[1]
    data = train_nid, val_nid, test_nid, in_feats, n_classes, g

    # Train and evaluate.
    epoch_time, test_acc = run(args, device, data)
    print(
        f"Summary of node classification(GraphSAGE): GraphName "
        f"{args.graph_name} | TrainEpochTime(mean) {epoch_time:.4f} "
        f"| TestAccuracy {test_acc:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed GraphSAGE.")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument(
        "--ip_config", type=str, help="The file for IP configuration"
    )
    parser.add_argument(
        "--part_config", type=str, help="The path to the partition config file"
    )
    parser.add_argument(
        "--n_classes", type=int, default=0, help="the number of classes"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        help="pytorch distributed backend",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="the number of GPU device. Use 0 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="10,25")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--local_rank", type=int, help="get rank of the process"
    )
    parser.add_argument(
        "--pad-data",
        default=False,
        action="store_true",
        help="Pad train nid to the same length across machine, to ensure num "
        "of batches to be the same.",
    )
    parser.add_argument(
        "--use_graphbolt",
        action="store_true",
        help="Use GraphBolt for distributed train.",
    )
    args = parser.parse_args()
    print(f"Arguments: {args}")
    main(args)
