#!/bin/bash
#SBATCH -p gpu-debug
#SBATCH -A c01062
#SBATCH --nodes=2
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --job-name=gs_lp
#SBATCH --output=/N/slate/yuzih/DGL/DistTraining/logs/load-job-%j.out
#SBATCH --error=/N/slate/yuzih/DGL/DistTraining/logs/load-job-%j.err
#SBATCH --time=01:00:00
# -------------------------
source /N/slate/yuzih/miniconda3/etc/profile.d/conda.sh
conda activate dgl
module load cudatoolkit/12.2

DATASET="yelp"
DATAPATH="/N/slate/yuzih/DGL/DistTraining/dataset/rawdata"
OUTPUT="/N/slate/yuzih/DGL/DistTraining/dataset/graph/yelp"
SCRIPT="/N/slate/yuzih/DGL/DistTraining/dataloading/more_data.py"

python $SCRIPT \
    --dataset $DATASET \
    --datapath $DATAPATH \
    --output $OUTPUT

