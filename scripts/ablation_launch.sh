#!/bin/bash
source ~/slurm-env.sh 2>/dev/null
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate hf2vad
echo "### setup ablation trees ###"
bash ~/setup_ablation.sh memae 2>&1 | tail -2
bash ~/setup_ablation.sh 2mem 2>&1 | tail -2
cd ~
AM=$(sbatch --parsable --job-name=ablmemae --export=ALL,TREE=$HOME/hf2vad_ablate_memae ~/hf2vad_ds_train.sbatch)
A2=$(sbatch --parsable --job-name=abl2mem --export=ALL,TREE=$HOME/hf2vad_ablate_2mem ~/hf2vad_ds_train.sbatch)
echo "### ABLATION TRAIN: memae=$AM 2mem=$A2 ###"
declare -A CK CF
CK[ped2]=$HOME/scratch/checkpoints/hf2vad/ped2_ML_MemAE_SC_CVAE_finetune/best.pth
CF[ped2]=$HOME/hf2vad/cfgs/finetune_cfg.yaml
CK[avenue]=$HOME/scratch/checkpoints/hf2vad_avenue/avenue_ML_MemAE_SC_CVAE_finetune/best.pth
CF[avenue]=$HOME/hf2vad_avenue/cfgs/finetune_cfg.yaml
CK[shanghaitech]=$HOME/scratch/checkpoints/hf2vad_shanghaitech/shanghaitech_ML_MemAE_SC_CVAE_finetune/best.pth
CF[shanghaitech]=$HOME/hf2vad_shanghaitech/cfgs/finetune_cfg.yaml
echo "### ckpt existence ###"
for ds in ped2 avenue shanghaitech; do [ -s "${CK[$ds]}" ] && echo "$ds ckpt OK" || echo "$ds ckpt MISSING"; done
for ds in ped2 avenue shanghaitech; do
  for mode in flow frame; do
    if [ "$mode" = flow ]; then WR=1; WP=0; else WR=0; WP=1; fi
    J=$(sbatch --parsable --job-name=sc_${ds}_${mode} --export=ALL,CKPT=${CK[$ds]},CFG=${CF[$ds]},WR=$WR,WP=$WP,EXP=${ds}_score_${mode} ~/hf2vad_score_eval.sbatch)
    echo "SCORE $ds $mode -> $J"
  done
done
echo "### queue ###"; squeue -u $USER -o "%.9i %.14j %.9T %.8M" | head -20
