#!/bin/bash
#SBATCH --account=gzmcagent
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/xchenhs/scratch/runs/hf2vad/oeval_%x_%j.out
#SBATCH --error=/home/xchenhs/scratch/runs/hf2vad/oeval_%x_%j.out
set -e
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate hf2vad
export PYTHONIOENCODING=utf-8
: "${CKPT:?need CKPT}"; : "${CFG:?need CFG}"
export H=$HOME/hf2vad
cd $H
ln -sfn ~/scratch/data/hf2vad/pretrained_ckpts $H/pretrained_ckpts 2>/dev/null || true
mkdir -p $H/eval
LOCALCFG=$H/official_eval_cfg.yaml
cp "$CFG" "$LOCALCFG"
python - <<'PY'
import yaml, os
p=os.path.join(os.environ["H"],"official_eval_cfg.yaml")
c=yaml.safe_load(open(p))
c["dataset_base_dir"]="./data"; c["eval_root"]="./eval"; c["device"]="cuda:0"; c["num_workers"]=4
yaml.safe_dump(c, open(p,"w"))
print("=== official eval:", c["dataset_name"], "| w_r", c.get("w_r"), "| w_p", c.get("w_p"), "| exp", c.get("exp_name"))
PY
echo "===== OFFICIAL EVAL $CKPT ====="
PYTHONPATH=$H python eval.py --model_save_path "$CKPT" --cfg_file "$LOCALCFG"
echo "===== OFFICIAL EVAL DONE ====="
