#!/bin/bash
#SBATCH --account=gzmcagent
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/xchenhs/scratch/runs/hf2vad/ppstc_%x_%j.out
#SBATCH --error=/home/xchenhs/scratch/runs/hf2vad/ppstc_%x_%j.out
set -e
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate hf2vad
export PYTHONIOENCODING=utf-8
: "${DS:?need DS}"
export H=$HOME/hf2vad
echo "===== STC $DS on $(hostname) ====="
cd $H
PYTHONPATH=$H python pre_process/extract_samples.py --proj_root $H --dataset_name $DS --mode train
PYTHONPATH=$H python pre_process/extract_samples.py --proj_root $H --dataset_name $DS --mode test
echo "train chunks: $(ls $H/data/$DS/training/chunked_samples/|wc -l)  test chunks: $(ls $H/data/$DS/testing/chunked_samples/|wc -l)"
echo "===== STC_DONE $DS ====="
