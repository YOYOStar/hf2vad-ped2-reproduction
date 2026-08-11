import os, time, yaml, torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from models.mem_cvae import HFVAD
from datasets.dataset import Chunked_sample_dataset

H = os.path.expanduser("~/hf2vad")
OUT = os.path.expanduser("~/scratch/runs/hf2vad/extras"); os.makedirs(OUT, exist_ok=True)
dev = "cuda:0"
cfg = yaml.safe_load(open(os.path.join(H, "cfgs/finetune_cfg.yaml")))
mp = cfg["model_paras"]
model = HFVAD(num_hist=mp["clip_hist"], num_pred=mp["clip_pred"], config=cfg,
              features_root=mp["feature_root"], num_slots=mp["num_slots"],
              shrink_thres=mp["shrink_thres"], mem_usage=mp["mem_usage"], skip_ops=mp["skip_ops"]).to(dev).eval()
ck = os.path.expanduser("~/scratch/checkpoints/hf2vad/ped2_ML_MemAE_SC_CVAE_finetune/best.pth")
model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"])

ds = Chunked_sample_dataset(os.path.join(H, "data/ped2/testing/chunked_samples/chunked_samples_00.pkl"))
dl = DataLoader(ds, batch_size=128, num_workers=4, shuffle=False)
sf_func = nn.MSELoss(reduction="none")

# ---------- timing (inference) ----------
b = next(iter(dl)); sf = b[0].to(dev); so = b[1].to(dev)
with torch.no_grad():
    for _ in range(3): model(sf, so, mode="test")
torch.cuda.synchronize(); t0 = time.time(); N = 30
with torch.no_grad():
    for _ in range(N): model(sf, so, mode="test")
torch.cuda.synchronize(); dt = (time.time() - t0) / N
nc = sf.shape[0]
print("TIMING inference: %.2f ms/batch(%d cubes) = %.3f ms/cube" % (dt*1000, nc, dt*1000/nc))

# ---------- per-cube scores over all test cubes + stash frames ----------
scores = []; store = []
with torch.no_grad():
    for sf, so, bb, pf, idx in dl:
        sf = sf.to(dev); so = so.to(dev)
        out = model(sf, so, mode="test")
        fe = sf_func(out["frame_pred"], out["frame_target"]).cpu().numpy()
        oe = sf_func(out["of_recon"], out["of_target"]).cpu().numpy()
        fs = fe.reshape(fe.shape[0], -1).sum(1); osr = oe.reshape(oe.shape[0], -1).sum(1)
        sc = cfg["w_r"]*osr + cfg["w_p"]*fs
        ft = out["frame_target"].cpu().numpy(); fp = out["frame_pred"].cpu().numpy()
        for i in range(len(sc)):
            scores.append(float(sc[i])); store.append((ft[i], fp[i], fe[i].sum(0)))
scores = np.array(scores)
print("test cubes:", len(scores), "score min/median/max: %.1f/%.1f/%.1f" % (scores.min(), np.median(scores), scores.max()))

# ---------- Fig.6-style difference maps: normal (low) + abnormal (high) ----------
order = np.argsort(scores)
picks = [("normal", order[len(order)//20]), ("normal", order[len(order)//10]),
         ("abnormal", order[-1]), ("abnormal", order[-30])]
fig, ax = plt.subplots(len(picks), 3, figsize=(7.5, 2.4*len(picks)))
for r, (tag, k) in enumerate(picks):
    ft, fp, err = store[k]
    gt = np.transpose(ft[-3:], (1, 2, 0)); gt = (gt - gt.min())/(np.ptp(gt)+1e-9)
    pr = np.transpose(fp[-3:], (1, 2, 0)); pr = (pr - pr.min())/(np.ptp(pr)+1e-9)
    ax[r, 0].imshow(gt[..., ::-1]); ax[r, 0].set_ylabel("%s\nscore=%.0f" % (tag, scores[k]), fontsize=8)
    ax[r, 1].imshow(pr[..., ::-1])
    ax[r, 2].imshow(err, cmap="jet")
    for c in range(3): ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    if r == 0:
        ax[r, 0].set_title("GT frame", fontsize=9); ax[r, 1].set_title("HF2-VAD pred", fontsize=9); ax[r, 2].set_title("pred error", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig6_diff_maps_ped2.png"), dpi=130); plt.close()
print("saved fig6_diff_maps_ped2.png")
print("EXTRAS_DONE")
