#!/bin/bash
#SBATCH --account=gzmcagent
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/xchenhs/scratch/runs/hf2vad/ppflow_%x_%j.out
#SBATCH --error=/home/xchenhs/scratch/runs/hf2vad/ppflow_%x_%j.out
set -e
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate hf2vad
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda12.2/toolkit/12.2.2 2>/dev/null || true
export PYTHONIOENCODING=utf-8
: "${DS:?need DS}"
export H=$HOME/hf2vad
echo "===== FLOW $DS on $(hostname) ====="
nvidia-smi --query-gpu=name --format=csv,noheader
cd $H/pre_process
PYTHONPATH=$H python extract_flows.py --proj_root $H --dataset_name $DS --mode train
PYTHONPATH=$H python extract_flows.py --proj_root $H --dataset_name $DS --mode test
echo "train flows: $(find $H/data/$DS/training/flows -name '*.npy'|wc -l)  test flows: $(find $H/data/$DS/testing/flows -name '*.npy'|wc -l)"
echo "===== FLOW_DONE $DS ====="
