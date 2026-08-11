#!/bin/bash
source ~/slurm-env.sh 2>/dev/null
cd ~
SJID=$(sbatch --parsable --gres=gpu:1 --qos=normal_qos hf2vad_stc.sbatch)
echo "===== STC_SUBMITTED $SJID ====="
for i in $(seq 1 120); do
  if ! squeue -u "$USER" -h -o %i | grep -q "$SJID"; then break; fi
  sleep 15
done
echo "===== STC_DONE $SJID ====="
tail -22 ~/scratch/runs/hf2vad/stc_${SJID}.out
CK=~/hf2vad/data/ped2/testing/chunked_samples/chunked_samples_00.pkl
CKT=~/hf2vad/data/ped2/training/chunked_samples/chunked_samples_00.pkl
if [ -s "$CK" ] && [ -s "$CKT" ]; then
  EJID=$(sbatch --parsable hf2vad_eval.sbatch)
  echo "===== EVAL_SUBMITTED $EJID ====="
  for i in $(seq 1 80); do
    if ! squeue -u "$USER" -h -o %i | grep -q "$EJID"; then break; fi
    sleep 15
  done
  echo "===== EVAL_DONE $EJID ====="
  tail -40 ~/scratch/runs/hf2vad/eval_${EJID}.out
else
  echo "===== STC_FAILED: chunked_samples missing, eval not submitted ====="
fi
echo "===== ORCH2_END ====="
