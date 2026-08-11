import pickle, os, numpy as np, shutil

SH = os.path.expanduser("~/scratch/data/hf2vad/assets/shtech_kaggle/SHANGHAI")
DATA = os.path.expanduser("~/scratch/data/hf2vad/data/shanghaitech")
GT = os.path.join(DATA, "ground_truth_demo", "gt_label.json")

gt = pickle.load(open(GT, "rb"))
test_clips = set(k[:-4] if k.endswith(".npy") else k for k in gt.keys())
print("expected test clips (gt keys):", len(test_clips))

src_dirs = [os.path.join(SH, "SHANGHAI_Test", "frames"),
            os.path.join(SH, "SHANGHAI_TRAIN", "frames")]
clip_src = {}
for sd in src_dirs:
    if os.path.isdir(sd):
        for c in sorted(os.listdir(sd)):
            p = os.path.join(sd, c)
            if os.path.isdir(p):
                clip_src[c] = p   # later dir wins on dup (shouldn't dup)
print("total clips found in kaggle dataset:", len(clip_src))

test_root = os.path.join(DATA, "testing", "frames")
train_root = os.path.join(DATA, "training", "frames")
for r in (test_root, train_root):
    if os.path.isdir(r):
        shutil.rmtree(r)
    os.makedirs(r)

ntest = ntrain = 0
for c, src in clip_src.items():
    dst = os.path.join(test_root if c in test_clips else train_root, c)
    os.symlink(src, dst)
    if c in test_clips:
        ntest += 1
    else:
        ntrain += 1

missing = test_clips - set(clip_src.keys())
print("symlinked  test=%d  train=%d  missing_test=%d %s" % (ntest, ntrain, len(missing), list(missing)[:10]))

# verify test frame counts vs gt (sorted-key order == eval order)
mism = []
for k in sorted(gt.keys()):
    c = k[:-4]
    fp = os.path.join(test_root, c)
    n = len([x for x in os.listdir(fp) if x.endswith(".jpg")]) if os.path.isdir(fp) else -1
    exp = len(np.asarray(gt[k]).reshape(-1))
    if n != exp:
        mism.append((c, n, exp))
print("test frame-count mismatches vs gt:", len(mism), mism[:10])
print("DONE")
