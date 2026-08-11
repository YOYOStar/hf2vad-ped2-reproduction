"""Pixel-level comparison: author JPG ped2 vs original UCSD tif ped2 (shared data tree)."""
import cv2
import glob
import os
import sys
import numpy as np

AUTHOR = os.path.expanduser(sys.argv[1])   # .../data_author/ped2
ORIG = os.path.expanduser(sys.argv[2])     # .../data/ped2 (shared, tif)

for split in ("training", "testing"):
    a_root = os.path.join(AUTHOR, split, "frames")
    o_root = os.path.join(ORIG, split, "frames")
    a_vids = sorted(d for d in glob.glob(os.path.join(a_root, "*")) if os.path.isdir(d))
    o_vids = sorted(d for d in glob.glob(os.path.join(o_root, "*")) if os.path.isdir(d))
    assert len(a_vids) == len(o_vids), f"{split}: video count {len(a_vids)} vs {len(o_vids)}"
    diffs, worst = [], (0.0, "")
    n_frames = 0
    for av, ov in zip(a_vids, o_vids):
        af = sorted(glob.glob(os.path.join(av, "*.jpg")))
        of = sorted(glob.glob(os.path.join(ov, "*.tif")))
        assert len(af) == len(of), f"{av}: {len(af)} vs {ov}: {len(of)}"
        for a, o in zip(af, of):
            ia = cv2.imread(a).astype(np.float32)
            io = cv2.imread(o).astype(np.float32)
            assert ia.shape == io.shape, f"shape {ia.shape} vs {io.shape} @ {a}"
            d = np.abs(ia - io)
            m = float(d.mean())
            diffs.append(m)
            if m > worst[0]:
                worst = (m, f"{os.path.basename(av)}/{os.path.basename(a)} maxpix={d.max():.0f}")
            n_frames += 1
    diffs = np.array(diffs)
    print(f"[{split}] frames={n_frames} meanAbsDiff={diffs.mean():.3f} "
          f"p95={np.percentile(diffs,95):.3f} worstFrame={worst[1]} ({worst[0]:.3f})")
print("COMPARE_DONE")
