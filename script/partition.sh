#!/bin/bash
#SBATCH -p gpu-debug
#SBATCH -A c01062
#SBATCH --nodes=2
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --job-name=gs_lp
#SBATCH --output=/N/slate/yuzih/Graphstorm/logs/job-%j.out
#SBATCH --error=/N/slate/yuzih/Graphstorm/logs/job-%j.err
#SBATCH --time=01:00:00
# -------------------------
source /N/slate/yuzih/miniconda3/etc/profile.d/conda.sh
conda activate dgl
module load cudatoolkit/12.2

DATASET="ogbn-products"
GRAPHPATH="/N/slate/yuzih/DGL/1DistTraining/dataset/graph/ogbn-products"
OUTPUT="/N/slate/yuzih/DGL/1DistTraining/dataset/parted_graph/ogbn-products"
SCRIPT="/N/slate/yuzih/DGL/1DistTraining/partitioning/graph_partition.py"

python $SCRIPT \
    --graph_path $GRAPHPATH \
    --dataset $DATASET \
    --num_parts 2 \
    --output $OUTPUT \
    --balance_train

