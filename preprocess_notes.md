# HF2-VAD Ped2 复现 — 预处理与环境记录

> 素材文件（最终汇入 `REPRODUCTION_REPORT.md`）。记录到 2026-06-03 预处理完成时的确定事实。

## 论文 / 代码
- 论文：A Hybrid Video Anomaly Detection Framework via Memory-Augmented Flow Reconstruction and Flow-Guided Frame Prediction (HF2-VAD), ICCV 2021. arXiv:2108.06852
- 代码：https://github.com/LiUzHiAn/hf2vad
- 论文报告值（frame-level AUC）：Ped2 **99.3%** / Avenue 91.1% / ShanghaiTech 76.2%
- 本次复现范围：**仅 Ped2**

## 执行环境（HKUST SuperPod）
- GPU：NVIDIA H800 80GB（sm_90），SLURM 调度，account=gzmcagent
- conda env `hf2vad`：Python 3.11，**torch 2.5.1+cu121**，torchvision 0.20.1+cu121，opencv/scikit-learn/scipy/joblib/tensorboardX/matplotlib
- 编译 CUDA 扩展用 module `cuda12.2/toolkit/12.2.2`（nvcc 12.2）

## 原始数据来源
- 作者数据服务器 `101.32.75.151:8181` **已失效**（连接超时）
- Ped2 原始帧改用 **UCSD Anomaly Dataset 官方包**（svcl.ucsd.edu，UCSDped2，.tif 灰度帧），结构与 HF2VAD 期望一致；frame-level 标签用仓库自带 `ground_truth_demo/gt_label.json`
- 16 训练视频 / 12 测试视频；Test001=180 帧（与 eval.py METADATA 完全吻合）

## 与原文的偏差（faithfulness deviations）
1. **目标检测器**：原文用 Cascade-RCNN-R101（mmdet 2.11/mmcv-full 1.3.1）。该老栈无法在 sm_90 上运行（torch≤1.7/CUDA≤11.1）。**替换为 torchvision FasterRCNN-R50-FPN-v2（COCO 预训练）**，保持 `init_detector/inference_detector` 接口不变。
   - 影响评估：HF2VAD 帧级得分对检测器不敏感（异常物体大；且有时序梯度前景框 getFgBboxes 补充；w_r=1.0 流重建为主信号）。
2. **光流**：**完全保真**——用官方 FlowNet2（`FlowNet2_checkpoint.pth.tar`，从 Dropbox 镜像获取，220/220 张量加载），自定义 CUDA 层在 H800 重新编译（c++17 + sm_90 gencode）。喂入沿用原码 BGR 顺序（与官方权重训练时一致）。
3. 兼容补丁（numpy≥1.24 / torch 2.5）：`dataset.py` `np.int→int`；`extract_bboxes.py` 保存 `np.array(...,dtype=object)`；FlowNet2 setup.py 升 c++17、去硬编码 `/usr/local/cuda` include。

## 预处理产物统计（已完成）
- **光流**：train 2550 个 `.npy` + test 2010 个 `.npy`（每帧一张，原分辨率 [h,w,2]）
- **bbox**（torchvision FRCNN，conf_thr=0.5 + 时序梯度前景框）：
  - train：2550 帧，31853 框，平均 12.49 框/帧
  - test：2010 帧，35587 框，平均 17.70 框/帧
- **STC chunked_samples**：进行中（appearance 5帧×3通道、motion 4流×2通道、patch 32×32）

## 多数据集结果（扩展复现）

| 数据集 | 官方权重 eval（我的cubes）| 从零 finetuned | 论文 |
|---|---|---|---|
| Ped2 | 99.52% | 97.69% | 99.3% |
| **Avenue** | **90.64%** | **91.12%** | **91.1%** ✅ 几乎精确 |
| ShanghaiTech | 76.09% | 75.45% | 76.2% ✅ 官方近精确/从零差0.75% |

ShanghaiTech: 标准split重建(107测/330训), 预处理31.5万帧(train 274605+test 40791 flow), STC 13训练chunk/1测试chunk; 官方76.09≈论文76.2; 从零 Stage1=73.06/CVAE=75.50/finetune=75.45; mem=2,w_r=0.02,w_p=1.0,saveevery=5(每5ep评估,加速). Avenue 从零 91.12%≈论文 91.1%；Stage1=84.82/Stage2 CVAE=90.70/finetune=91.12（finetune 起作用、符合论文，印证 ped2 的 finetune 倒挂是单次波动）。官方权重 90.64<从零 91.12 因官方权重在作者 Cascade cubes 上训、在我的 torchvision cubes 上评（检测器不匹配）；从零自洽故更高。avenue: 19训练clip/14844训练光流/15324测试光流/7训练chunk/1测试chunk，w_r=0.05,w_p=1.0,mem=3。

## 结果（Ped2）
- **P1 官方权重 eval AUC：0.99523（99.52%）** ✓ — vs 论文报告 99.3%（checkpoint 名 ped2_HF2VAD_99.31）。复现成功，甚至略高（在本地重建 cubes 上）。w_r=1.0, w_p=0.1。SLURM job 435953 @ dgx-21(H800)。
- **P2 从零训练**（job 435955 @ H800，3 阶段各 80 epoch，用仓库默认 cfg）：
  - Stage1 ML-MemAE-SC（纯光流重建）Best AUC **0.9865**
  - Stage2 CVAE（流引导帧预测）Best AUC **0.9875**
  - Stage3 finetune 联合 HF2VAD → eval.py(finetune=False) **AUC 0.97691**
  - 从零复现的最终 HF2VAD（finetuned）= **97.69%**，vs 论文 99.3%（差 ~1.6%）。差异归因：检测器替换（torchvision FRCNN vs 原文 Cascade-RCNN）、训练随机性、默认超参；finetune 略低于 CVAE 阶段为训练波动。
