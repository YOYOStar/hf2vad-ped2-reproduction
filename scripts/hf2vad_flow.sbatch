#!/bin/bash
#SBATCH --job-name=hf2vad-flow
#SBATCH --account=gzmcagent
#SBATCH --partition=normal
#SBATCH --qos=normal_debug_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/home/xchenhs/scratch/runs/hf2vad/flow_%j.out
#SBATCH --error=/home/xchenhs/scratch/runs/hf2vad/flow_%j.out
set -e
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate hf2vad
source /etc/profile.d/modules.sh 2>/dev/null || source /etc/profile.d/lmod.sh 2>/dev/null || true
module load cuda12.2/toolkit/12.2.2 2>/dev/null || true
export TORCH_CUDA_ARCH_LIST="9.0"
export PYTHONIOENCODING=utf-8
echo "===== job $SLURM_JOB_ID on $(hostname) ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "nvcc: $(which nvcc 2>/dev/null)"; nvcc --version 2>/dev/null | tail -2; echo "CUDA_HOME=${CUDA_HOME:-unset}"
H=$HOME/hf2vad
echo "===== build FlowNet2 custom CUDA layers ====="
for pkg in correlation_package resample2d_package channelnorm_package; do
  echo "----- building $pkg -----"
  cd $H/pre_process/flownet_networks/$pkg
  rm -rf build *.egg-info dist
  pip install --no-build-isolation -q . 2>&1 | tail -6
done
echo "===== sanity: import ops + FlowNet2 fwd + load ckpt ====="
cd $H
python - <<'PY'
import torch, numpy as np
import correlation_cuda, resample2d_cuda, channelnorm_cuda
print("ops imported OK")
from pre_process.flownet_networks.flownet2_models import FlowNet2
m = FlowNet2().cuda().eval()
ck = torch.load('pre_process/assets/FlowNet2_checkpoint.pth.tar', map_location='cpu', weights_only=False)['state_dict']
md = m.state_dict(); pd = {k:v for k,v in ck.items() if k in md}; md.update(pd); m.load_state_dict(md)
print("FlowNet2 loaded params %d/%d" % (len(pd), len(md)))
x = (torch.rand(1,3,2,768,1024).cuda())*255.0
with torch.no_grad(): f = m(x)
print("flow output:", tuple(f.shape), "mean", float(f.float().mean()))
PY
echo "===== extract flows: TRAIN ====="
cd $H/pre_process
PYTHONPATH=$H python extract_flows.py --proj_root $H --dataset_name ped2 --mode train
echo "===== extract flows: TEST ====="
PYTHONPATH=$H python extract_flows.py --proj_root $H --dataset_name ped2 --mode test
echo "===== flow .npy counts ====="
echo "train flows: $(find $H/data/ped2/training/flows -name '*.npy' 2>/dev/null | wc -l)"
echo "test  flows: $(find $H/data/ped2/testing/flows  -name '*.npy' 2>/dev/null | wc -l)"
echo "===== DONE ====="
