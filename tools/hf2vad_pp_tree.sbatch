#!/bin/bash
#SBATCH --account=gzmcagent
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/xchenhs/scratch/runs/hf2vad/pp_%x_%j.out
#SBATCH --error=/home/xchenhs/scratch/runs/hf2vad/pp_%x_%j.out
set -e
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate hf2vad
source /etc/profile.d/modules.sh 2>/dev/null || source /etc/profile.d/lmod.sh 2>/dev/null || true
module load cuda12.2/toolkit/12.2.2 2>/dev/null || true
export PYTHONIOENCODING=utf-8
: "${DS:?need DS env}"
: "${TREE:?need TREE env}"
export H=$TREE
echo "===== preprocess $DS tree=$TREE on $(hostname) job=$SLURM_JOB_ID ====="
nvidia-smi --query-gpu=name --format=csv,noheader
cd $H/pre_process
echo "######## BBOX $DS (torchvision FRCNN) ########"
PYTHONPATH=$H python extract_bboxes.py --proj_root $H --dataset_name $DS --mode train
PYTHONPATH=$H python extract_bboxes.py --proj_root $H --dataset_name $DS --mode test
echo "######## FLOW $DS (FlowNet2) ########"
PYTHONPATH=$H python extract_flows.py --proj_root $H --dataset_name $DS --mode train
PYTHONPATH=$H python extract_flows.py --proj_root $H --dataset_name $DS --mode test
echo "######## STC $DS ########"
cd $H
PYTHONPATH=$H python pre_process/extract_samples.py --proj_root $H --dataset_name $DS --mode train
PYTHONPATH=$H python pre_process/extract_samples.py --proj_root $H --dataset_name $DS --mode test
echo "######## SUMMARY $DS ########"
ls -l $H/data/$DS/${DS}_bboxes_train.npy $H/data/$DS/${DS}_bboxes_test.npy 2>&1
echo "train flows: $(find $H/data/$DS/training/flows -name '*.npy' 2>/dev/null | wc -l)"
echo "test  flows: $(find $H/data/$DS/testing/flows  -name '*.npy' 2>/dev/null | wc -l)"
echo "train chunks: $(ls $H/data/$DS/training/chunked_samples/ 2>/dev/null | wc -l)"
echo "test  chunks: $(ls $H/data/$DS/testing/chunked_samples/ 2>/dev/null | wc -l)"
echo "######## PREPROCESS DONE $DS ########"
