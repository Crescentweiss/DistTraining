import argparse
import dgl
import dgl.distributed

def partition_graph(
    graph,
    dataset,
    num_parts,
    output,
    part_method="metis",
    balance_train=False,
    balance_edges=False,
    num_trainers_per_machine=1,
    use_graphbolt=False
):
    balance_ntypes = graph.ndata["train_mask"] if balance_train else None

    dgl.distributed.partition_graph(
        graph,
        dataset,
        num_parts,
        output,
        part_method=part_method,
        
        balance_ntypes=balance_ntypes,
        balance_edges=balance_edges,
        num_trainers_per_machine=num_trainers_per_machine,
        use_graphbolt=use_graphbolt,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Partition a saved DGL graph")
    parser.add_argument("--graph_path", type=str, required=True, help="Path to the saved DGL graph .bin")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (for naming outputs)")
    parser.add_argument("--num_parts", type=int, default=4, help="Number of partitions")
    parser.add_argument("--output", type=str, default="parted", help="Output folder path")
    parser.add_argument("--part_method", type=str, default="metis", help="Partition method")
    parser.add_argument("--balance_train", action="store_true", help="Balance training set")
    parser.add_argument("--balance_edges", action="store_true", help="Balance edge count")
    parser.add_argument("--num_trainers_per_machine", type=int, default=1)
    parser.add_argument("--use_graphbolt", action="store_true")
    args = parser.parse_args()

    g = dgl.load_graphs(args.graph_path)[0][0]
    partition_graph(
        g,
        args.dataset,
        args.num_parts,
        args.output,
        args.part_method,
        args.balance_train,
        args.balance_edges,
        args.num_trainers_per_machine,
        args.use_graphbolt,
    )
