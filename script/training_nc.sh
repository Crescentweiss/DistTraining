#!/bin/bash
#SBATCH -p gpu-debug
#SBATCH -A c01062
#SBATCH --nodes=2
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --job-name=gs_lp
#SBATCH --output=/N/slate/yuzih/DGL/DistTraining/logs/job-%j.out
#SBATCH --error=/N/slate/yuzih/DGL/DistTraining/logs/job-%j.err
#SBATCH --time=01:00:00
# -------------------------
echo "Job started at $(date)"

echo "Running on nodes:"
scontrol show hostnames $SLURM_JOB_NODELIST

CONDA_SOURCE="/N/slate/yuzih/miniconda3/etc/profile.d/conda.sh"
source $CONDA_SOURCE
conda activate dgl
module load cudatoolkit/12.2

IP_CONFIG="/N/slate/yuzih/DGL/DistTraining/script/ip_config.txt"
scontrol show hostnames $SLURM_JOB_NODELIST | while read nodename; do
  getent hosts "$nodename" | awk '{print $1}'
done > "$IP_CONFIG"

echo "IP config:"
cat "$IP_CONFIG"

DATASET="ogbn-products"
PART_CONFIG="/N/slate/yuzih/DGL/DistTraining/dataset/parted_graph/ogbn-products/ogbn-products.json"
WORKSPACE="/N/slate/yuzih/DGL"
PYTHON_SCRIPT="/N/slate/yuzih/DGL/DistTraining/training/node_classification.py"

echo "Starting distributed training..."
python3 $WORKSPACE/dgl/tools/launch.py \
  --workspace $WORKSPACE/dgl/examples/distributed/graphsage/ \
  --num_trainers 2 \
  --num_samplers 0 \
  --num_servers 1 \
  --part_config $PART_CONFIG \
  --ip_config $IP_CONFIG \
  "source $CONDA_SOURCE && \
  conda activate dgl && \
  python3 $PYTHON_SCRIPT \
      --graph_name $DATASET \
      --ip_config $IP_CONFIG \
      --part_config $PART_CONFIG \
      --backend gloo \
      --num_gpus 2 \
      --num_epochs 10 \
      --num_hidden 16 \
      --batch_size 1000"

echo "Job finished at $(date)"