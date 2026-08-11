#!/bin/bash
source ~/slurm-env.sh 2>/dev/null
SEEDS="1 2 3 4 5"
for s in $SEEDS; do
  T=/home/xchenhs/hf2vad_s$s
  rsync -a --delete \
    --exclude='ckpt' --exclude='log' --exclude='eval' \
    --exclude='*/build' --exclude='*.egg-info' --exclude='*/dist' --exclude='__pycache__' \
    ~/hf2vad/ "$T/"
  mkdir -p ~/scratch/checkpoints/hf2vad_s$s "$T/log" "$T/eval"
  ln -sfn ~/scratch/checkpoints/hf2vad_s$s "$T/ckpt"
  jid=$(sbatch --parsable --job-name=hf2vad-s$s --export=ALL,HF2VAD_SEED=$s,TREE=$T ~/seed_train.sbatch)
  echo "submitted seed $s -> job $jid  (tree $T, data->$(readlink $T/data))"
done
echo "===== ALL 5 SEEDS SUBMITTED ====="
squeue -u "$USER" -o "%.10i %.14j %.10T %.8M %R"
