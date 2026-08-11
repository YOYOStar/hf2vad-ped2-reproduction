#!/bin/bash
# Wait for bbox job, then submit+wait STC. Args: $1 = bbox job id
source ~/slurm-env.sh 2>/dev/null
BJID="$1"
echo "waiting for bbox job $BJID ..."
for i in $(seq 1 80); do
  if ! squeue -u "$USER" -h -o %i | grep -q "$BJID"; then break; fi
  sleep 15
done
echo "===== BBOX_DONE ====="
grep -aE "frames:|saved|Traceback|Error|rror" ~/scratch/runs/hf2vad/bbox_${BJID}.out | tail -8
ls -l ~/hf2vad/data/ped2/ped2_bboxes_train.npy ~/hf2vad/data/ped2/ped2_bboxes_test.npy 2>/dev/null
if [ -s ~/hf2vad/data/ped2/ped2_bboxes_train.npy ] && [ -s ~/hf2vad/data/ped2/ped2_bboxes_test.npy ]; then
  cd ~
  SJID=$(sbatch --parsable --qos=normal_qos hf2vad_stc.sbatch)
  echo "===== STC_SUBMITTED $SJID ====="
  for i in $(seq 1 120); do
    if ! squeue -u "$USER" -h -o %i | grep -q "$SJID"; then break; fi
    sleep 15
  done
  echo "===== STC_DONE $SJID ====="
  tail -24 ~/scratch/runs/hf2vad/stc_${SJID}.out
else
  echo "===== BBOX_FAILED: bbox npy empty, STC not submitted ====="
fi
echo "===== ORCHESTRATION_END ====="
