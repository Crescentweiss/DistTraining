import argparse
import time

import dgl
import torch as th
from ogb.nodeproppred import DglNodePropPredDataset

def load_ogb(name, root):
    """Load ogbn dataset."""
    data = DglNodePropPredDataset(name=name, root=root)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata["features"] = graph.ndata.pop("feat")
    graph.ndata["labels"] = labels
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
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
        default="reddit",
        help="datasets: reddit, ogbn-products, ogbn-papers100M",
    )
    argparser.add_argument(
        "--datapath",
        type=str,
        default="reddit",
        help="data path: /N/slate/yuzih/DGL/1DistTraining/dataset",
    )
    argparser.add_argument(
        "--output", 
        type=str, 
        default="graph.bin", 
        help="Output binary file name",
    )
    args = argparser.parse_args()

    start = time.time()
    if args.dataset in ["ogbn-products", "ogbn-papers100M"]:
        g, _ = load_ogb(args.dataset, root=args.datapath)
    else:
        raise RuntimeError(f"Unknown dataset: {args.dataset}")
    print(
        "Load {} takes {:.3f} seconds".format(args.dataset, time.time() - start)
    )

    dgl.save_graphs(args.output, g)
    print("|V|={}, |E|={}".format(g.num_nodes(), g.num_edges()))
