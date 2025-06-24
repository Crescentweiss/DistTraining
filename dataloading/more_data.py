import argparse
import time

import dgl
import torch as th
from dgl.data import (
    RedditDataset,
    YelpDataset,
)

def load_dataset(name, self_loop=True):
    """Load common node classification datasets."""
    if name == "reddit":
        data = RedditDataset(self_loop=self_loop)
        g = data[0]
        g.ndata["features"] = g.ndata.pop("feat")
        g.ndata["labels"] = g.ndata.pop("label").long()
        return g, data.num_classes

    elif name == "yelp":
        data = YelpDataset()
        g = data[0]
        g.ndata["features"] = g.ndata.pop("feat")
        # 原始多标签：[N, C]
        multi_label = g.ndata.pop("label").float()
        # 找到标签不全为 0 的节点（即至少属于一个类别）
        valid_mask = multi_label.sum(dim=1) > 0
        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        # 创建子图，只保留 valid 节点
        g = dgl.node_subgraph(g, valid_idx)
        # 将多标签转换为单标签（主类别）
        single_label = multi_label[valid_idx].argmax(dim=1).long()
        g.ndata["labels"] = single_label

        return g, multi_label.shape[1]
    
    else:
        raise ValueError(f"Unknown dataset: {name}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition graph")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="reddit",
        help="datasets: reddit, cora, citeseer, pubmed, amazon-computer, amazon-photo",
    )
    argparser.add_argument(
        "--datapath",
        type=str,
        default="reddit",
        help="data path: /N/slate/yuzih/DGL/DistTraining/dataset",
    )
    argparser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output path of partitioned graph.",
    )
    args = argparser.parse_args()

    start = time.time()
    if args.dataset == "reddit":
        g, _ = load_dataset("reddit", self_loop=True)
    elif args.dataset == "yelp":
        g, _ = load_dataset("yelp",)
    else:
        raise RuntimeError(f"Unknown dataset: {args.dataset}")
    print(
        "Load {} takes {:.3f} seconds".format(args.dataset, time.time() - start)
    )
    
    dgl.save_graphs(args.output, g)
    print("|V|={}, |E|={}".format(g.num_nodes(), g.num_edges()))