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
        multi_label = g.ndata.pop("label").float()

        # 只保留有标签的节点
        valid_mask = multi_label.sum(dim=1) > 0
        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        g = dgl.node_subgraph(g, valid_idx)

        # 多标签转为单标签
        single_label = multi_label[valid_idx].argmax(dim=1).long()
        g.ndata["labels"] = single_label

        # 保存类别数（可选：供调试打印用）
        n_classes = single_label.max().item() + 1
        print(f"Yelp processed into single-label with {n_classes} classes")

        return g, n_classes
    
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
        g, num_classes = load_dataset("yelp")
    else:
        raise RuntimeError(f"Unknown dataset: {args.dataset}")
    print(
        "Load {} takes {:.3f} seconds".format(args.dataset, time.time() - start)
    )
    
    dgl.save_graphs(args.output, g)
    print("|V|={}, |E|={}".format(g.num_nodes(), g.num_edges()))
    print(f"num_classes = {num_classes}")