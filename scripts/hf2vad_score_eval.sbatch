#!/bin/bash
#SBATCH --account=gzmcagent
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/xchenhs/scratch/runs/hf2vad/score_%x_%j.out
#SBATCH --error=/home/xchenhs/scratch/runs/hf2vad/score_%x_%j.out
set -e
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate hf2vad
export PYTHONIOENCODING=utf-8
: "${CKPT:?}"; : "${CFG:?}"; : "${WR:?}"; : "${WP:?}"; : "${EXP:?}"
export H=$HOME/hf2vad
cd $H
mkdir -p $H/eval
LOCALCFG=$H/score_${EXP}.yaml
cp "$CFG" "$LOCALCFG"
python - <<'PY'
import yaml, os
H=os.environ["H"]; p=os.path.join(H,"score_%s.yaml"%os.environ["EXP"])
c=yaml.safe_load(open(p))
c["dataset_base_dir"]="./data"; c["eval_root"]="./eval"; c["device"]="cuda:0"; c["num_workers"]=4
c["w_r"]=float(os.environ["WR"]); c["w_p"]=float(os.environ["WP"]); c["exp_name"]=os.environ["EXP"]
yaml.safe_dump(c, open(p,"w"))
print("score-eval ds=%s exp=%s w_r=%s w_p=%s" % (c["dataset_name"], c["exp_name"], c["w_r"], c["w_p"]))
PY
echo "===== SCORE EVAL $EXP (w_r=$WR w_p=$WP) ckpt=$CKPT ====="
PYTHONPATH=$H python eval.py --model_save_path "$CKPT" --cfg_file "$LOCALCFG"
echo "===== SCORE_EVAL_DONE $EXP ====="
