import argparse
import time

import dgl
import torch as th
from ogb.nodeproppred import DglNodePropPredDataset

def load_ogb(name, root, sample_ratio=0.1):
    """Load ogbn dataset, support partial subgraph extraction for large graphs."""
    data = DglNodePropPredDataset(name=name, root=root)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata["features"] = graph.ndata.pop("feat")
    graph.ndata["labels"] = labels
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    if name == "ogbn-papers100M":
        # ⚠️ 提取中间一段节点，构造子图
        num_nodes = graph.num_nodes()
        sample_num = int(num_nodes * sample_ratio)
        start = num_nodes // 2 - sample_num // 2
        end = start + sample_num
        sample_nodes = th.arange(start, end)

        print(f"[INFO] Extracting subgraph with nodes {start} to {end} (total {sample_num})")
        graph = dgl.node_subgraph(graph, sample_nodes)
        labels = graph.ndata["labels"]

        # 重新计算划分（从原始 split 中选出仍在子图内的）
        node_ids = graph.ndata[dgl.NID]  # 原始 ID
        orig_train, orig_val, orig_test = (
            set(splitted_idx["train"].tolist()),
            set(splitted_idx["valid"].tolist()),
            set(splitted_idx["test"].tolist()),
        )

        local_train = [i for i, nid in enumerate(node_ids) if nid.item() in orig_train]
        local_val   = [i for i, nid in enumerate(node_ids) if nid.item() in orig_val]
        local_test  = [i for i, nid in enumerate(node_ids) if nid.item() in orig_test]

        train_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
        val_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
        test_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)

        train_mask[local_train] = True
        val_mask[local_val] = True
        test_mask[local_test] = True

        graph.ndata["train_mask"] = train_mask
        graph.ndata["val_mask"] = val_mask
        graph.ndata["test_mask"] = test_mask
        return graph, num_labels

    else:
        # 原始处理逻辑
        train_nid, val_nid, test_nid = (
            splitted_idx["train"],
            splitted_idx["valid"],
            splitted_idx["test"],
        )
        train_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
        train_mask[train_nid] = True
        val_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
        val_mask[val_nid] = True
        test_mask = th.zeros((graph.num_nodes(),), dtype=th.bool)
        test_mask[test_nid] = True
        graph.ndata["train_mask"] = train_mask
        graph.ndata["val_mask"] = val_mask
        graph.ndata["test_mask"] = test_mask
        return graph, num_labels

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition graph")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="datasets: ogbn-products, ogbn-papers100M",
    )
    argparser.add_argument(
        "--datapath",
        type=str,
        default="ogbn-products",
        help="data path: /N/slate/yuzih/DGL/DistTraining/dataset",
    )
    argparser.add_argument(
        "--output", 
        type=str, 
        default="graph.bin", 
        help="Output binary file name",
    )
    args = argparser.parse_args()

    start = time.time()
    if args.dataset in ["ogbn-products","ogbn-arxiv", "ogbn-papers100M"]:
        g, _ = load_ogb(args.dataset, root=args.datapath)
    else:
        raise RuntimeError(f"Unknown dataset: {args.dataset}")
    print(
        "Load {} takes {:.3f} seconds".format(args.dataset, time.time() - start)
    )

    dgl.save_graphs(args.output, g)
    print("|V|={}, |E|={}".format(g.num_nodes(), g.num_edges()))
