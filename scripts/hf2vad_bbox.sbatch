#!/bin/bash
#SBATCH --job-name=hf2vad-bbox
#SBATCH --account=gzmcagent
#SBATCH --partition=normal
#SBATCH --qos=normal_debug_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/home/xchenhs/scratch/runs/hf2vad/bbox_%j.out
#SBATCH --error=/home/xchenhs/scratch/runs/hf2vad/bbox_%j.out
set -e
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate hf2vad
export PYTHONIOENCODING=utf-8
echo "===== job $SLURM_JOB_ID on $(hostname) ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export H=$HOME/hf2vad
cd $H/pre_process
echo "===== extract bboxes: TRAIN (torchvision FRCNN-R50-FPN-v2) ====="
PYTHONPATH=$H python extract_bboxes.py --proj_root $H --dataset_name ped2 --mode train
echo "===== extract bboxes: TEST ====="
PYTHONPATH=$H python extract_bboxes.py --proj_root $H --dataset_name ped2 --mode test
echo "===== bbox files ====="
ls -l $H/data/ped2/ped2_bboxes_train.npy $H/data/ped2/ped2_bboxes_test.npy
python - <<'PY'
import numpy as np, os
H=os.environ["H"]
for m in ["train","test"]:
    b=np.load(os.path.join(H,"data/ped2/ped2_bboxes_%s.npy"%m), allow_pickle=True)
    cnts=[len(x) for x in b]
    print(m, "frames:", len(b), "total boxes:", int(np.sum(cnts)), "avg/frame: %.2f"%(np.mean(cnts)))
PY
echo "===== DONE ====="
