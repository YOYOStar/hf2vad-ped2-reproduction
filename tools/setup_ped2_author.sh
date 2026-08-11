#!/bin/bash
# Set up ped2 author-data control experiment: extract pack, pixel-compare,
# build tree ~/hf2vad_ped2_author (jpg patch) + tree ~/hf2vad_ped2_orig2 (seed control).
set -e
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate hf2vad

A=~/scratch/data/hf2vad/data_author
SHARED=~/scratch/data/hf2vad/data

echo "======== 1. EXTRACT ========"
mkdir -p "$A"
if [ ! -d "$A/ped2_extracted" ]; then
  mkdir -p "$A/ped2_extracted"
  tar -xzf ~/scratch/data/hf2vad/assets/ped2_authorpack.tar.gz -C "$A/ped2_extracted"
fi
P=$(find "$A/ped2_extracted" -maxdepth 4 -type d -path '*training/frames' | head -1)
P=$(dirname "$(dirname "$P")")
echo "author ped2 root: $P"
ln -sfn "$P" "$A/ped2"
echo "train videos: $(ls "$A/ped2/training/frames" | wc -l), test videos: $(ls "$A/ped2/testing/frames" | wc -l)"

echo "======== 2. PIXEL COMPARE (jpg vs tif) ========"
python ~/compare_ped2.py "$A/ped2" "$SHARED/ped2"

echo "======== 3. TREE ped2_author ========"
T=~/hf2vad_ped2_author
rsync -a --delete \
  --exclude='ckpt' --exclude='log' --exclude='eval' \
  --exclude='*/build' --exclude='*.egg-info' --exclude='*/dist' --exclude='__pycache__' \
  ~/hf2vad/ "$T/"
rm -f "$T/data"
mkdir -p "$T/data" "$T/log" "$T/eval"
ln -sfn "$A/ped2" "$T/data/ped2"
cp -rn "$SHARED/ped2/ground_truth_demo" "$A/ped2/" 2>/dev/null || true
mkdir -p ~/scratch/checkpoints/hf2vad_ped2_author
ln -sfn ~/scratch/checkpoints/hf2vad_ped2_author "$T/ckpt"
ln -sfn ~/scratch/data/hf2vad/pretrained_ckpts "$T/pretrained_ckpts"
# ensure assets (flownet/detector weights) resolve
[ -e "$T/assets/FlowNet2_checkpoint.pth.tar" ] || ln -sfn ~/scratch/data/hf2vad/assets/FlowNet2_checkpoint.pth.tar "$T/assets/FlowNet2_checkpoint.pth.tar"

echo "-- patch img_ext ped2 .tif -> .jpg --"
python - <<'PY'
import os
p = os.path.expanduser("~/hf2vad_ped2_author/datasets/dataset.py")
s = open(p).read()
old = '"ped2": ".tif"'
assert s.count(old) == 1, f"expected 1 occurrence, got {s.count(old)}"
open(p, "w").write(s.replace(old, '"ped2": ".jpg"'))
print("patched:", p)
PY
grep -n '"ped2":' "$T/datasets/dataset.py" | head -2

echo "======== 4. TREE ped2_orig2 (seed control, reuses shared preprocessed data) ========"
T2=~/hf2vad_ped2_orig2
rsync -a --delete \
  --exclude='ckpt' --exclude='log' --exclude='eval' \
  --exclude='*/build' --exclude='*.egg-info' --exclude='*/dist' --exclude='__pycache__' \
  ~/hf2vad/ "$T2/"
mkdir -p "$T2/log" "$T2/eval" ~/scratch/checkpoints/hf2vad_ped2_orig2
ln -sfn "$SHARED" "$T2/data"
ln -sfn ~/scratch/checkpoints/hf2vad_ped2_orig2 "$T2/ckpt"
ln -sfn ~/scratch/data/hf2vad/pretrained_ckpts "$T2/pretrained_ckpts"

echo "======== 5. SANITY: flownet ops importable ========"
python -c "import correlation_cuda, resample2d_cuda, channelnorm_cuda; print('flownet ops OK')" \
  || echo "WARN: flownet ops not importable on login node (may still work on compute node with cuda module)"

echo "SETUP_DONE"
