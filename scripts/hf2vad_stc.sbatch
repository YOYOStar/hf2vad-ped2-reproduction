#!/bin/bash
#SBATCH --job-name=hf2vad-stc
#SBATCH --account=gzmcagent
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/xchenhs/scratch/runs/hf2vad/stc_%j.out
#SBATCH --error=/home/xchenhs/scratch/runs/hf2vad/stc_%j.out
set -e
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate hf2vad
export PYTHONIOENCODING=utf-8
echo "===== job $SLURM_JOB_ID on $(hostname) ====="
export H=$HOME/hf2vad
cd $H
echo "===== extract STC samples: TRAIN ====="
PYTHONPATH=$H python pre_process/extract_samples.py --proj_root $H --dataset_name ped2 --mode train
echo "===== extract STC samples: TEST ====="
PYTHONPATH=$H python pre_process/extract_samples.py --proj_root $H --dataset_name ped2 --mode test
echo "===== chunked_samples ====="
ls -lh $H/data/ped2/training/chunked_samples/ $H/data/ped2/testing/chunked_samples/
python - <<'PY'
import joblib, os, numpy as np
H=os.environ["H"]
for m in ["training","testing"]:
    f=os.path.join(H,"data/ped2",m,"chunked_samples/chunked_samples_00.pkl")
    d=joblib.load(f)
    print(m, "samples:", len(d["sample_id"]),
          "| appearance", np.array(d["appearance"]).shape,
          "| motion", np.array(d["motion"]).shape)
PY
echo "===== DONE ====="
