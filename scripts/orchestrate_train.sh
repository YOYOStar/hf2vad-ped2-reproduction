#!/bin/bash
source ~/slurm-env.sh 2>/dev/null
cd ~
TJID=$(sbatch --parsable hf2vad_train.sbatch)
echo "===== TRAIN_SUBMITTED $TJID ====="
for i in $(seq 1 900); do
  if ! squeue -u "$USER" -h -o %i | grep -q "$TJID"; then break; fi
  sleep 20
done
echo "===== TRAIN_DONE $TJID ====="
LOG=~/scratch/runs/hf2vad/train_${TJID}.out
echo "--- stage markers / Best AUC / errors ---"
grep -aE "STAGE [123]|Best AUC|ALL TRAINING DONE|Traceback|Error:|rror |FAILED|CUDA out" "$LOG" | tail -40
echo "--- last 15 raw lines ---"
tail -15 "$LOG"
echo "--- best.pth ckpts ---"
find ~/hf2vad/ckpt -name best.pth -exec ls -lh {} \; 2>/dev/null
echo "===== ORCH_TRAIN_END ====="
