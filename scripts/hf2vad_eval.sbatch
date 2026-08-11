#!/bin/bash
#SBATCH --job-name=hf2vad-eval
#SBATCH --account=gzmcagent
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/xchenhs/scratch/runs/hf2vad/eval_%j.out
#SBATCH --error=/home/xchenhs/scratch/runs/hf2vad/eval_%j.out
set -e
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate hf2vad
export PYTHONIOENCODING=utf-8
echo "===== job $SLURM_JOB_ID on $(hostname) ====="
nvidia-smi --query-gpu=name --format=csv,noheader
export H=$HOME/hf2vad
cd $H
ln -sfn ~/scratch/data/hf2vad/pretrained_ckpts $H/pretrained_ckpts
mkdir -p $H/eval
LOCALCFG=$H/ped2_eval_cfg.yaml
cp $H/pretrained_ckpts/ped2_HF2VAD_99.31_cfg.yaml $LOCALCFG
python - <<'PY'
import yaml, os
H=os.environ["H"]; p=os.path.join(H,"ped2_eval_cfg.yaml")
c=yaml.safe_load(open(p))
c["dataset_base_dir"]="./data"; c["eval_root"]="./eval"; c["device"]="cuda:0"; c["num_workers"]=4
yaml.safe_dump(c, open(p,"w"))
print("=== eval cfg ===")
print("dataset:", c["dataset_name"], "| base:", c["dataset_base_dir"], "| w_r:", c.get("w_r"), "| w_p:", c.get("w_p"), "| exp:", c.get("exp_name"))
PY
echo "===== RUN EVAL (official ped2_HF2VAD_99.31.pth) ====="
PYTHONPATH=$H python eval.py \
  --model_save_path $H/pretrained_ckpts/ped2_HF2VAD_99.31.pth \
  --cfg_file $LOCALCFG
echo "===== EVAL DONE ====="
