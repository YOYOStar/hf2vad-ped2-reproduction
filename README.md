# HF2-VAD Reproduction (Ped2 / Avenue / ShanghaiTech)

Full reproduction of **HF2-VAD** (ICCV 2021, [arXiv:2108.06852](https://arxiv.org/abs/2108.06852), [official code](https://github.com/LiUzHiAn/hf2vad)) on modern hardware: **PyTorch 2.5.1 + CUDA 12.1, NVIDIA H800 (sm_90), SLURM**.

## Headline results (frame-level AUC %)

| Dataset | Paper | Official-weights eval (our pipeline) | From-scratch (finetuned) |
|---|---|---|---|
| UCSD Ped2 | 99.3 | **99.52** | 97.69 – 98.67 (4 runs) |
| CUHK Avenue | 91.1 | **90.64** | **91.12** |
| ShanghaiTech | 76.2 | **76.09** | 75.45 |

All three headline results reproduce within normal no-fixed-seed variance. See [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md) for the full report, including:

- **§4–5** Engineering patches for torch 2.5 / sm_90 (FlowNet2 CUDA ops recompile, C++17, numpy ≥1.24 fixes) and the single substantive deviation (Cascade R-CNN → torchvision FasterRCNN-v2 detector, mmdet 2.x cannot run on sm_90).
- **§11** Ablations reproducing paper Table 1 / Table 2 trends, qualitative figures, and timing.
- **§13** Controlled data-provenance experiment (2026-08): the widely circulated JPG repack of Ped2 vs. the original UCSD tifs — pixel diff 0.65/255, result diff 0.02 AUC. Also documents that the **finetune stage consistently shows a small *negative* gain on Ped2 (4/4 independent runs)** while helping on Avenue — the paper's stage-3 contribution is dataset-dependent.

## Repo layout

- `REPRODUCTION_REPORT.md` — full report (Chinese), including cross-model (GPT) adversarial review notes.
- `preprocess_notes.md` — dataset acquisition & preprocessing notes.
- `figures/` — anomaly score curves (per test video), ROC, Fig.6-style prediction-difference maps.
- `scripts/` — SLURM sbatch files and orchestration scripts (preprocess / 3-stage train / eval / ablations / multi-seed).
- `tools/` — data-provenance control experiment scripts (§13): pixel comparison, author-pack setup, tree-parameterized sbatch variants.

## Notes for reproducers

1. Ped2 frames must be `.tif` with `Train*/Test*` folder names — the official code hardcodes both (`datasets/dataset.py`); the circulated JPG repack (`training/frames/01/*.jpg`) will silently enumerate **zero** frames.
2. FlowNet2 official checkpoint: author's link is dead; a working Dropbox mirror is referenced in the report. Custom CUDA ops need `-std=c++17` and `sm_90` gencode on H800.
3. Best-epoch selection uses **test-set AUC** (upstream repo protocol, kept for comparability) — numbers across this literature are inflated by this; treat sub-0.5 AUC gaps on Ped2 as noise.

## License

MIT (this repo's scripts and report). The original HF2-VAD code remains under its upstream license.
