#!/bin/bash
#SBATCH --account=gzmcagent
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-23:00:00
#SBATCH --output=/home/xchenhs/scratch/runs/hf2vad/dstrain_%x_%j.out
#SBATCH --error=/home/xchenhs/scratch/runs/hf2vad/dstrain_%x_%j.out
set -e
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate hf2vad
export PYTHONIOENCODING=utf-8
: "${TREE:?need TREE env}"
export HF2VAD_SEED=${HF2VAD_SEED:-2021}
cd "$TREE"
echo "===== ds-train tree=$TREE seed=$HF2VAD_SEED host=$(hostname) job=$SLURM_JOB_ID ====="
nvidia-smi --query-gpu=name --format=csv,noheader
mkdir -p "$TREE/log" "$TREE/eval"
echo "######## STAGE1 ML-MemAE-SC ########"
PYTHONPATH=$TREE python ml_memAE_sc_train.py
echo "######## STAGE2 CVAE ########"
PYTHONPATH=$TREE python train.py
echo "######## STAGE3 finetune ########"
PYTHONPATH=$TREE python finetune.py
FT_EXP=$(python -c "import yaml;print(yaml.safe_load(open('$TREE/cfgs/finetune_cfg.yaml'))['exp_name'])")
LOCALCFG=$TREE/selftrain_eval_cfg.yaml
cp $TREE/cfgs/finetune_cfg.yaml $LOCALCFG
python - <<'PY'
import yaml, os
H=os.environ["TREE"]; p=os.path.join(H,"selftrain_eval_cfg.yaml")
c=yaml.safe_load(open(p)); c["dataset_base_dir"]="./data"; c["eval_root"]="./eval"; c["device"]="cuda:0"; c["num_workers"]=4
yaml.safe_dump(c, open(p,"w"))
PY
echo "######## FINAL EVAL $FT_EXP ########"
PYTHONPATH=$TREE python eval.py --model_save_path $TREE/ckpt/$FT_EXP/best.pth --cfg_file $LOCALCFG
echo "######## DS-TRAIN DONE (tree=$TREE) ########"
