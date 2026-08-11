#!/bin/bash
# Set up a per-dataset working tree ~/hf2vad_<DS> with dataset-specific cfgs.
DS="$1"
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate hf2vad
T="$HOME/hf2vad_$DS"
rsync -a --delete \
  --exclude='ckpt' --exclude='log' --exclude='eval' \
  --exclude='*/build' --exclude='*.egg-info' --exclude='*/dist' --exclude='__pycache__' \
  ~/hf2vad/ "$T/"
ln -sfn ~/scratch/data/hf2vad/data "$T/data"
mkdir -p ~/scratch/checkpoints/hf2vad_$DS "$T/log" "$T/eval"
ln -sfn ~/scratch/checkpoints/hf2vad_$DS "$T/ckpt"
python ~/make_cfgs.py "$DS" ~/hf2vad/cfgs "$T/cfgs"
echo "=== $DS cfgs (key fields) ==="
grep -H "dataset_name:\|exp_name:\|w_r:\|w_p:\|num_epochs:\|mem_usage:" "$T"/cfgs/*.yaml
echo "=== tree $T ready (data->$(readlink $T/data)  ckpt->$(readlink $T/ckpt)) ==="
