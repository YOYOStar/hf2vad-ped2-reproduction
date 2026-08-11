#!/bin/bash
#SBATCH --account=gzmcagent
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/xchenhs/scratch/runs/hf2vad/seed_%x_%j.out
#SBATCH --error=/home/xchenhs/scratch/runs/hf2vad/seed_%x_%j.out
set -e
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate hf2vad
export PYTHONIOENCODING=utf-8
: "${TREE:?need TREE}"; : "${HF2VAD_SEED:?need HF2VAD_SEED}"
cd "$TREE"
echo "===== SEED=$HF2VAD_SEED TREE=$TREE host=$(hostname) job=$SLURM_JOB_ID ====="
nvidia-smi --query-gpu=name --format=csv,noheader
echo "##### STAGE1 ML-MemAE-SC (seed $HF2VAD_SEED) #####"
PYTHONPATH=$TREE python ml_memAE_sc_train.py
echo "##### STAGE2 CVAE (seed $HF2VAD_SEED) #####"
PYTHONPATH=$TREE python train.py
echo "##### STAGE3 finetune (seed $HF2VAD_SEED) #####"
PYTHONPATH=$TREE python finetune.py
echo "##### FINAL EVAL (seed $HF2VAD_SEED) #####"
LOCALCFG=$TREE/selftrain_eval_cfg.yaml
cp $TREE/cfgs/finetune_cfg.yaml $LOCALCFG
python - <<'PY'
import yaml, os
H=os.environ["TREE"]; p=os.path.join(H,"selftrain_eval_cfg.yaml")
c=yaml.safe_load(open(p)); c["dataset_base_dir"]="./data"; c["eval_root"]="./eval"; c["device"]="cuda:0"; c["num_workers"]=4
yaml.safe_dump(c, open(p,"w"))
PY
FINAUC=$(PYTHONPATH=$TREE python eval.py --model_save_path $TREE/ckpt/ped2_ML_MemAE_SC_CVAE_finetune/best.pth --cfg_file $LOCALCFG 2>/dev/null | tail -1)
echo "SEEDRESULT seed=$HF2VAD_SEED finetuned_auc=$FINAUC"
echo "===== SEED $HF2VAD_SEED DONE ====="
