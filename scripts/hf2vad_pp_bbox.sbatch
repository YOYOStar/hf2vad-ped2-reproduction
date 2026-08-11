#!/bin/bash
#SBATCH --account=gzmcagent
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/home/xchenhs/scratch/runs/hf2vad/ppbbox_%x_%j.out
#SBATCH --error=/home/xchenhs/scratch/runs/hf2vad/ppbbox_%x_%j.out
set -e
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate hf2vad
export PYTHONIOENCODING=utf-8
: "${DS:?need DS}"
export H=$HOME/hf2vad
echo "===== BBOX $DS on $(hostname) ====="
nvidia-smi --query-gpu=name --format=csv,noheader
cd $H/pre_process
PYTHONPATH=$H python extract_bboxes.py --proj_root $H --dataset_name $DS --mode train
PYTHONPATH=$H python extract_bboxes.py --proj_root $H --dataset_name $DS --mode test
ls -l $H/data/$DS/${DS}_bboxes_train.npy $H/data/$DS/${DS}_bboxes_test.npy
echo "===== BBOX_DONE $DS ====="
