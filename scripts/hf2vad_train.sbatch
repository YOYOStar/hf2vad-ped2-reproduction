#!/bin/bash
#SBATCH --job-name=hf2vad-train
#SBATCH --account=gzmcagent
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/home/xchenhs/scratch/runs/hf2vad/train_%j.out
#SBATCH --error=/home/xchenhs/scratch/runs/hf2vad/train_%j.out
set -e
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate hf2vad
export PYTHONIOENCODING=utf-8
echo "===== job $SLURM_JOB_ID on $(hostname) ====="
nvidia-smi --query-gpu=name --format=csv,noheader
export H=$HOME/hf2vad
cd $H
mkdir -p ~/scratch/checkpoints/hf2vad $H/log $H/eval
ln -sfn ~/scratch/checkpoints/hf2vad $H/ckpt
ln -sfn ~/scratch/data/hf2vad/pretrained_ckpts $H/pretrained_ckpts 2>/dev/null || true
echo "############ STAGE 1: ML-MemAE-SC (flow reconstruction) ############"
PYTHONPATH=$H python ml_memAE_sc_train.py
echo "############ STAGE 2: CVAE (flow-guided frame prediction) ############"
PYTHONPATH=$H python train.py
echo "############ STAGE 3: finetune (joint HF2VAD) ############"
PYTHONPATH=$H python finetune.py
echo "############ ALL TRAINING DONE ############"
echo "--- ckpt tree ---"; find $H/ckpt -name 'best.pth' -exec ls -lh {} \;
echo "############ FINAL EVAL: self-trained finetuned HF2VAD (eval.py, finetune=False) ############"
LOCALCFG=$H/selftrain_eval_cfg.yaml
cp $H/cfgs/finetune_cfg.yaml $LOCALCFG
python - <<'PY'
import yaml, os
H=os.environ["H"]; p=os.path.join(H,"selftrain_eval_cfg.yaml")
c=yaml.safe_load(open(p))
c["dataset_base_dir"]="./data"; c["eval_root"]="./eval"; c["device"]="cuda:0"; c["num_workers"]=4
yaml.safe_dump(c, open(p,"w"))
PY
PYTHONPATH=$H python eval.py \
  --model_save_path $H/ckpt/ped2_ML_MemAE_SC_CVAE_finetune/best.pth \
  --cfg_file $LOCALCFG
echo "############ SELF-TRAINED FINAL AUC PRINTED ABOVE ############"
