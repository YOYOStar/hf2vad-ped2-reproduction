import cv2, os

RAW = os.path.expanduser("~/scratch/data/hf2vad/assets/avenue_raw/Avenue Dataset")
OUT = os.path.expanduser("~/scratch/data/hf2vad/data/avenue")
# eval.py METADATA avenue testing_frames_cnt (21 videos) -- must match for gt alignment
EXPECT_TEST = [1439, 1211, 923, 947, 1007, 1283, 605, 36, 1175, 841,
               472, 1271, 549, 507, 1001, 740, 426, 294, 248, 273, 76]
# Avenue19 training cuts: 1-indexed inclusive ranges to EXCLUDE (paper sec. "Notice - Avenue")
CUTS = {2: [(311, 521), (771, 831)], 4: [(1460, 1510)], 7: [(741, 900)]}
JPGQ = 95


def extract(vpath, out_root, vid, exclude=None):
    exclude = exclude or []
    cap = cv2.VideoCapture(vpath)
    def ex(k):
        return any(a <= k <= b for a, b in exclude)
    seg = 0
    prev_ex = False
    counts = {}
    k = 0
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        k += 1
        if ex(k):
            prev_ex = True
            continue
        if prev_ex:
            seg += 1
            prev_ex = False
        folder = ("%02d" % vid) if not exclude else ("%02d_%d" % (vid, seg))
        d = os.path.join(out_root, folder)
        os.makedirs(d, exist_ok=True)
        counts[folder] = counts.get(folder, 0) + 1
        cv2.imwrite(os.path.join(d, "%05d.jpg" % counts[folder]), fr,
                    [cv2.IMWRITE_JPEG_QUALITY, JPGQ])
    cap.release()
    return k, counts


print("===== TEST extraction (21 videos) =====", flush=True)
test_counts = []
for v in range(1, 22):
    vp = os.path.join(RAW, "testing_videos", "%02d.avi" % v)
    total, _ = extract(vp, os.path.join(OUT, "testing", "frames"), v, exclude=None)
    test_counts.append(total)
print("test counts:", test_counts)
print("expected   :", EXPECT_TEST)
print("TEST_MATCH:", test_counts == EXPECT_TEST, flush=True)

print("===== TRAIN extraction (Avenue19 cuts) =====", flush=True)
for v in range(1, 17):
    vp = os.path.join(RAW, "training_videos", "%02d.avi" % v)
    total, counts = extract(vp, os.path.join(OUT, "training", "frames"), v, exclude=CUTS.get(v))
    if v in CUTS:
        print(" train v%02d total=%d -> %s" % (v, total, counts))
folders = sorted(os.listdir(os.path.join(OUT, "training", "frames")))
print("train folders (%d): %s" % (len(folders), folders), flush=True)
print("===== DONE =====")
