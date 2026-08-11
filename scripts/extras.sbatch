#!/bin/bash
#SBATCH --account=gzmcagent
#SBATCH --partition=normal
#SBATCH --qos=normal_debug_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/xchenhs/scratch/runs/hf2vad/extras_%j.out
#SBATCH --error=/home/xchenhs/scratch/runs/hf2vad/extras_%j.out
set -e
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate hf2vad
export PYTHONIOENCODING=utf-8
export H=$HOME/hf2vad
cd $H
nvidia-smi --query-gpu=name --format=csv,noheader
PYTHONPATH=$H python ~/extras.py
echo "===== EXTRAS SBATCH DONE ====="
