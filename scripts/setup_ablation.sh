#!/bin/bash
# Create a Ped2 ablation tree with modified ML-MemAE-SC memory config.
# usage: setup_ablation.sh <memae|2mem>
V="$1"
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate hf2vad
T="$HOME/hf2vad_ablate_$V"
rsync -a --delete \
  --exclude='ckpt' --exclude='log' --exclude='eval' \
  --exclude='*/build' --exclude='*.egg-info' --exclude='*/dist' --exclude='__pycache__' \
  ~/hf2vad/ "$T/"
ln -sfn ~/scratch/data/hf2vad/data "$T/data"
mkdir -p ~/scratch/checkpoints/hf2vad_ablate_$V "$T/log" "$T/eval"
ln -sfn ~/scratch/checkpoints/hf2vad_ablate_$V "$T/ckpt"
python - "$V" "$T" <<'PY'
import yaml, os, sys
V, T = sys.argv[1], sys.argv[2]
MEM = {"memae": [False, False, False, True], "2mem": [False, False, True, True]}[V]
SKIP = {"memae": ["none", "none", "none"], "2mem": ["none", "none", "concat"]}[V]
for f in ["ml_memAE_sc_cfg.yaml", "cfg.yaml", "finetune_cfg.yaml"]:
    p = os.path.join(T, "cfgs", f)
    c = yaml.safe_load(open(p))
    c["model_paras"]["mem_usage"] = MEM
    c["model_paras"]["skip_ops"] = SKIP
    c["num_workers"] = 8
    yaml.safe_dump(c, open(p, "w"), sort_keys=False)
print("variant", V, "mem_usage", MEM, "skip_ops", SKIP)
PY
echo "ablation tree $T ready (ckpt->$(readlink $T/ckpt))"
