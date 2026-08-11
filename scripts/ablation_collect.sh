#!/bin/bash
source ~/slurm-env.sh 2>/dev/null
PAT="439956|439957|439958|439959|439960|439961|439962|439963"
while squeue -u "$USER" -h -o %i 2>/dev/null | grep -qE "^($PAT)$"; do sleep 30; done
echo "===== ALL ABLATION/SCORING JOBS DONE ====="
echo "--- SCORING-BASED ABLATION (single-branch AUC from finetuned model) ---"
for ds in ped2 avenue shanghaitech; do
  for mode in flow frame; do
    L=$(ls -t ~/scratch/runs/hf2vad/score_sc_${ds}_${mode}_*.out 2>/dev/null | head -1)
    A=$(grep -aE "^0\.[0-9]+$" "$L" 2>/dev/null | tail -1)
    echo "  $ds  ${mode}-only  AUC=$A"
  done
done
echo "--- PED2 ABLATION TABLE-2 variants (MemAE / 2mem) ---"
for v in ablmemae abl2mem; do
  L=$(ls -t ~/scratch/runs/hf2vad/dstrain_${v}_*.out 2>/dev/null | head -1)
  echo "  == $v ($L) =="
  grep -aE "STAGE[123]|Best AUC|FINAL EVAL|^0\.[0-9]+$|DS-TRAIN DONE|Traceback|Error" "$L" 2>/dev/null | tail -10
done
echo "===== ABLATION_COLLECT_END ====="
