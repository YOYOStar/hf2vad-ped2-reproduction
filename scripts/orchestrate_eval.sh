#!/bin/bash
source ~/slurm-env.sh 2>/dev/null
cd ~
EJID=$(sbatch --parsable hf2vad_eval.sbatch)
echo "===== EVAL_SUBMITTED $EJID ====="
for i in $(seq 1 80); do
  if ! squeue -u "$USER" -h -o %i | grep -q "$EJID"; then break; fi
  sleep 15
done
echo "===== EVAL_DONE $EJID ====="
tail -45 ~/scratch/runs/hf2vad/eval_${EJID}.out
