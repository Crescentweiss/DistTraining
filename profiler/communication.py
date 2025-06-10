import argparse
import socket
import time
import logging

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

def analyze_cross_node_comm(g, input_nodes, blocks, trainers_per_node):
    """Analyze if a mini-batch requires cross-node communication and estimate communication volume"""
    
    # Get partition information
    pb = g.get_partition_book()
    current_partition = pb.partid  # 当前进程实际负责的分区
    
    # 使用当前分区作为本地分区进行比较
    current_node_partitions = {current_partition}
    
    # Count partition distribution of source nodes
    input_nids_np = input_nodes.cpu().numpy()
    node_locations = {}
    remote_nodes = 0
    
    # 分别收集本地和远程节点ID
    local_node_ids = []
    remote_node_ids = []
    
    # Iterate through each input node and determine its location
    for nid in input_nids_np:
        part_id = pb.nid2partid(nid).item()
        if part_id not in node_locations:
            node_locations[part_id] = 0
        node_locations[part_id] += 1
        
        # 判断是否需要跨节点通信
        if part_id not in current_node_partitions:
            remote_nodes += 1
            remote_node_ids.append(nid)
        else:
            local_node_ids.append(nid)

    # Estimate communication volume (assuming feature dimension is in_feats)
    in_feats = g.ndata['features'].shape[1]
    comm_bytes = remote_nodes * in_feats * 4  # Assuming float32 = 4 bytes
    
    return {
        'has_cross_node_comm': remote_nodes > 0,
        'remote_nodes': remote_nodes,
        'total_nodes': len(input_nids_np),
        'remote_ratio': remote_nodes / len(input_nids_np) if len(input_nids_np) > 0 else 0,
        'partition_distribution': node_locations,
        'estimated_comm_bytes': comm_bytes,
        'local_node_ids': th.tensor(local_node_ids),  # 本地节点ID
        'remote_node_ids': th.tensor(remote_node_ids)  # 远程节点ID
    }

def time_feature_access(g, local_node_ids, remote_node_ids):
    """测量本地和远程特征访问时间"""
    import time
    
    times = {
        'local_time': 0.0,
        'remote_time': 0.0,
        'local_count': len(local_node_ids),
        'remote_count': len(remote_node_ids)
    }
    
    # 测量本地特征访问时间
    if len(local_node_ids) > 0:
        start_time = time.time()
        local_features = g.ndata["features"][local_node_ids]
        local_features = local_features.cpu()  # 确保数据传输完成
        times['local_time'] = time.time() - start_time
    
    # 测量远程特征访问时间
    if len(remote_node_ids) > 0:
        start_time = time.time()
        remote_features = g.ndata["features"][remote_node_ids]
        remote_features = remote_features.cpu()  # 确保数据传输完成
        times['remote_time'] = time.time() - start_time
    
    return times

def analyze_node_degrees(g, input_nodes):
    """分析输入节点的度数分布"""
    degrees = g.in_degrees(input_nodes) + g.out_degrees(input_nodes)
    
    return {
        'min_degree': degrees.min().item(),
        'max_degree': degrees.max().item(), 
        'avg_degree': degrees.float().mean().item(),
        'total_edges': degrees.sum().item()
    }

def analyze_layer_comm(g, blocks):
    """Analyze communication requirements for each sampling layer"""
    
    layer_stats = []
    
    # Analyze each layer
    for i, block in enumerate(blocks):
        src_nodes = block.srcdata[dgl.NID]
        dst_nodes = block.dstdata[dgl.NID]
        
        # Analyze cross-node communication for this layer
        layer_comm = analyze_cross_node_comm(g, src_nodes, [block])
        layer_stats.append({
            'layer': i,
            'remote_ratio': layer_comm['remote_ratio'],
            'remote_nodes': layer_comm['remote_nodes'],
            'total_nodes': layer_comm['total_nodes'],
            'comm_bytes': layer_comm['estimated_comm_bytes']
        })
    
    return layer_stats

class CommunicationProfiler:
    """Communication profiler for DGL distributed training"""
    
    def __init__(self, g):
        """Initialize profiler
        
        Parameters
        ----------
        g : DistGraph
            The distributed graph
        """
        self.g = g
        self.reset_stats()
    
    def reset_stats(self):
        """Reset communication statistics"""
        self.comm_stats = {
            'total_batches': 0,
            'cross_node_batches': 0,
            'total_remote_nodes': 0,
            'total_nodes': 0,
            'total_comm_bytes': 0,
            'comm_time': 0.0,
            'per_partition_comm': {},
            'layer_stats': []
        }
    
    def profile_batch(self, input_nodes, seeds, blocks, fetch_time=None):
        """Profile communication for a batch
        
        Parameters
        ----------
        input_nodes : tensor
            Input nodes for the batch
        seeds : tensor
            Seed nodes for the batch
        blocks : list of DGLBlock
            List of computation blocks
        fetch_time : float, optional
            Feature fetching time if already measured
            
        Returns
        -------
        dict
            Communication analysis for this batch
        """
        # Analyze batch-level communication
        batch_comm = analyze_cross_node_comm(self.g, input_nodes, blocks)
        
        # Update statistics
        self.comm_stats['total_batches'] += 1
        if batch_comm['has_cross_node_comm']:
            self.comm_stats['cross_node_batches'] += 1
        
        self.comm_stats['total_remote_nodes'] += batch_comm['remote_nodes']
        self.comm_stats['total_nodes'] += batch_comm['total_nodes']
        self.comm_stats['total_comm_bytes'] += batch_comm['estimated_comm_bytes']
        
        # Record fetch time if provided
        if fetch_time is not None and batch_comm['has_cross_node_comm']:
            self.comm_stats['comm_time'] += fetch_time
        
        # Update per-partition communication
        for part_id, count in batch_comm['partition_distribution'].items():
            if part_id not in self.comm_stats['per_partition_comm']:
                self.comm_stats['per_partition_comm'][part_id] = 0
            self.comm_stats['per_partition_comm'][part_id] += count
        
        # Analyze layer-level communication
        layer_stats = analyze_layer_comm(self.g, blocks)
        self.comm_stats['layer_stats'].append(layer_stats)
        
        return batch_comm
    
    def report(self):
        """Generate communication profiling report
        
        Returns
        -------
        dict
            Detailed communication statistics
        """
        stats = self.comm_stats.copy()
        
        # Calculate additional statistics
        if stats['total_batches'] > 0:
            stats['cross_node_batch_ratio'] = stats['cross_node_batches'] / stats['total_batches']
        else:
            stats['cross_node_batch_ratio'] = 0
        
        if stats['total_nodes'] > 0:
            stats['remote_node_ratio'] = stats['total_remote_nodes'] / stats['total_nodes']
        else:
            stats['remote_node_ratio'] = 0
        
        # Calculate per-layer statistics
        if stats['layer_stats']:
            num_layers = len(stats['layer_stats'][0])
            layer_summary = []
            
            for layer in range(num_layers):
                layer_remote = sum(batch[layer]['remote_nodes'] for batch in stats['layer_stats'])
                layer_total = sum(batch[layer]['total_nodes'] for batch in stats['layer_stats'])
                layer_bytes = sum(batch[layer]['comm_bytes'] for batch in stats['layer_stats'])
                
                layer_summary.append({
                    'layer': layer,
                    'remote_nodes': layer_remote,
                    'total_nodes': layer_total,
                    'remote_ratio': layer_remote / layer_total if layer_total > 0 else 0,
                    'comm_bytes': layer_bytes
                })
            
            stats['layer_summary'] = layer_summary
        
        return stats

def print_comm_report(stats):
    """Print a formatted communication report
    
    Parameters
    ----------
    stats : dict
        Communication statistics from CommunicationProfiler.report()
    """
    print("\n=== Communication Profiling Report ===")
    print(f"Total batches: {stats['total_batches']}")
    print(f"Batches with cross-node communication: {stats['cross_node_batches']} ({stats['cross_node_batch_ratio']*100:.2f}%)")
    print(f"Total remote nodes: {stats['total_remote_nodes']} of {stats['total_nodes']} ({stats['remote_node_ratio']*100:.2f}%)")
    print(f"Total communication volume: {stats['total_comm_bytes']/1024/1024:.2f} MB")
    
    if stats['comm_time'] > 0:
        print(f"Communication time: {stats['comm_time']:.3f} seconds")
        print(f"Estimated bandwidth: {stats['total_comm_bytes']/stats['comm_time']/1024/1024:.2f} MB/s")
    
    if 'layer_summary' in stats:
        print("\n=== Layer Communication Statistics ===")
        for layer in stats['layer_summary']:
            print(f"Layer {layer['layer']}: {layer['remote_nodes']} remote nodes of {layer['total_nodes']} ({layer['remote_ratio']*100:.2f}%), {layer['comm_bytes']/1024/1024:.2f} MB")
    
    print("\n=== Per-Partition Communication ===")
    for part_id, count in stats['per_partition_comm'].items():
        print(f"Partition {part_id}: {count} nodes")