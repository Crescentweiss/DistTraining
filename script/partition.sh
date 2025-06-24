#!/bin/bash
#SBATCH -p gpu-debug
#SBATCH -A c01062
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --job-name=gs_lp
#SBATCH --output=/N/slate/yuzih/DGL/DistTraining/logs/job-%j.out
#SBATCH --error=/N/slate/yuzih/DGL/DistTraining/logs/job-%j.err
#SBATCH --time=01:00:00
# -------------------------
source /N/slate/yuzih/miniconda3/etc/profile.d/conda.sh
conda activate dgl
module load cudatoolkit/12.2

DATASET="yelp"
GRAPHPATH="/N/slate/yuzih/DGL/DistTraining/dataset/graph/yelp"
OUTPUT="/N/slate/yuzih/DGL/DistTraining/dataset/parted_graph/yelp"
SCRIPT="/N/slate/yuzih/DGL/DistTraining/partitioning/graph_partition.py"

python $SCRIPT \
    --graph_path $GRAPHPATH \
    --dataset $DATASET \
    --num_parts 2 \
    --output $OUTPUT \
    --balance_train

