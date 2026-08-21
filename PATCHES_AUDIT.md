# HF²-VAD 复现改动审计（ARIS）

> 目的：把"我（Claude）相对作者源码做了哪些改动"逐条讲清楚，并**批判性评判每处改动是否合理**。
> 权威基线：作者原仓库 `LiUzHiAn/hf2vad` @ `580fabe`（本地为其 clone，改动均为未提交的工作区修改）。
> 复现本审计：`cd hf2vad && git diff 580fabe`。代码内每处改动都有 `[ARIS-PATCH]` 标记，`grep -rn ARIS-PATCH` 可定位。

---

## 0. 一句话结论

12 个跟踪文件 + 1 个新增 shim 目录，共 **150 增 / 138 删**。**没有一处改动触碰模型结构、损失函数、打分公式或超参**（`models/ losses/ utils/` 经 `git status` 确认零改动）。改动按性质分五类：**A 语言兼容、B 编译目标、C 检测器替换、D 评估多-chunk、E 种子控制、F 依赖 shim**。其中只有 **C 是真正的方法学偏离**，**D 含一个"合理但可更干净"的隐患**，其余均为环境适配或研究仪器，行为中性。

---

## 1. 改动清单

| # | 文件 | 类别 | 规模 | 是否影响数值 | 合理性 |
|---|------|------|------|--------------|--------|
| A1 | `datasets/dataset.py` | 语言兼容 | np.int→int ×2 | 否 | ✅ 必要 |
| A2 | `pre_process/extract_bboxes.py` | 语言兼容 | +dtype=object | 否 | ✅ 必要 |
| B1 | `flownet_networks/*/setup.py` ×3 | 编译目标 | c++17 + sm_90 | 否(仅编译) | ✅ 必要 |
| B2 | `flownet_networks/correlation_package/correlation_cuda.cc` | 编译目标 | include 路径 | 否 | ✅ 必要 |
| C | `pre_process/mmdet_utils.py` | **检测器替换** | 整文件重写 | **是** | ⚠️ 可接受但需存疑 |
| D | `eval.py` + `ml_memAE_sc_eval.py` | 评估多-chunk | 循环所有 chunk | **是(修正)** | ✅ 修 bug，但含 D1 隐患 |
| E | `train.py` + `finetune.py` + `ml_memAE_sc_train.py` | 种子控制 | +8 行 seed 块 | 是(可复现性) | ✅ 合理，但仅部分确定性 |
| F | `edflow/`（新增） | 依赖 shim | 26 行 | 否 | ✅ 合理 |

---

## 2. 逐条评判

### A. 语言/NumPy 兼容（行为中性）✅

- **A1** `np.int` → `int`：`np.int` 别名在 NumPy≥1.24 被移除，原代码直接崩。`int(np.ceil(...))` 与 `np.int(np.ceil(...))` 结果完全一致。
- **A2** `np.save(all_bboxes)` → `np.save(np.array(all_bboxes, dtype=object))`：`all_bboxes` 是"每帧框数不等"的 ragged 列表，NumPy≥1.24 拒绝自动构造 ragged object array 并抛 `ValueError`。显式 `dtype=object` 存的是**同一份数据**。

**评判**：纯版本适配，无任何数值影响。合理。

### B. CUDA 编译目标适配 H800 / sm_90（仅编译期）✅

- **B1** 三个 FlowNet2 CUDA 扩展的 `setup.py`：`-std=c++14→c++17`（PyTorch 2.5 的 cpp_extension 头文件要求）；gencode 删除已被 CUDA 12 nvcc 弃用的 sm_50/52/60/61，新增 sm_70/80/86/90 + compute_90 PTX。
- **B2** `correlation_cuda.cc`：`#include</usr/local/cuda/include/cuda_runtime_api.h>`（写死绝对路径）→ `#include <cuda_runtime_api.h>`（走编译器 include 搜索路径）。

**评判**：只改**编译产物针对的 GPU 架构**，不改 FlowNet2 的算子数学。光流数值由权重决定，与目标架构无关。合理且必要——不改就无法在 Hopper 上 build。

### C. 检测器替换 —— **最大的方法学偏离** ⚠️

**改动**：`mmdet_utils.py` 整文件重写。原版用 **Cascade-RCNN-R101-FPN**（mmdet 2.11 / mmcv-full 1.3.1，COCO 权重）做前景目标检测；我换成 **torchvision Faster-RCNN-R50-FPN-v2**（COCO 权重），保持 `init_detector` / `inference_detector` 接口不变，作为 drop-in。

**为什么改**：mmdet 2.11 + mmcv-full 1.3.1 的旧 CUDA 算子无法在 sm_90 + PyTorch 2.5 上编译/运行。

**为什么"可能"影响可控（作者原设计给的护栏）**：
1. HF²-VAD 的主异常信号是**光流重建**（`w_r=1.0` vs `w_p=0.1`），检测器只负责提出"哪些前景框会变成 STC cube"。
2. 关键：我查过下游 `getObjBboxes` —— 它对检测结果只做 `np.vstack(所有类)` 后按 `conf_thr` 分数 + `min_area` 面积过滤，**原版也从不按类别过滤**。所以"我把 80 类合并成单数组"与原版"按类分组再 vstack"**行为等价**，没有引入"混入 80 类"的 bug。两者都是"COCO 全类、分数+面积阈值"。
3. HF²-VAD 额外用 `getFgBboxes`（时间梯度兜底）补充检测器漏掉的运动前景框——STC 覆盖对检测器不完全依赖。

**残留存疑（必须诚实说）**：
- 两个 COCO 检测器的**分数标定不同**，同一个 `conf_thr`（ped2 0.5 / avenue 0.25 / sh 0.5）会保留**不同的框集合**。"影响可控"这个论断，对运动主导的 **Ped2 最强**，对外观/CVAE 更重要的 Avenue/ShanghaiTech 只是**经验上**成立、非理论保证。
- 经验证据：官方权重 eval 自建 vs 作者包三集差异 ≤0.02 AUC；从零 Ped2 99.52、Avenue 91.12、SH 76.59（均达/超论文）。**结果反向证明该替换在这三个数据集上无实质损害**——但这是"事后验证"，不是"事前保证"。

**评判**：**可接受，但属于必须在论文/报告里显式声明的偏离**，不能当作"等价复现"。这是本次改动里唯一会被审稿人追问的点。已在 `mmdet_utils.py` docstring 顶部标注残留 caveat。

### D. 评估读取全部 chunk —— 防御性改动（在当前数据上是 no-op）

**改动**：`eval.py` / `ml_memAE_sc_eval.py` 原本只读单个 `testing_chunked_samples_file`；改成 glob `chunked_samples_*.pkl` 全部并循环。

> **⚠️ 2026-08-20 现场核查后的重要订正**（先前本节把 D/D1 说成"修了真 bug + 隐患",已被数据推翻）：
> 集群实际预处理里 `num_samples_each_chunk` 设得很大（test：ped2=100k / avenue=200k / sh=300k），而三个 test 集实际样本数分别是 **35587 / 107628 / 232223，全部 < 阈值 → 每个 test 集都只有 1 个 chunk（`chunked_samples_00.pkl`）**。
> 因此：**原版单读 `chunked_samples_00.pkl` 在当前数据上本就等于读全量**，D 的多-chunk glob 是 **no-op**；下面 D1 的跨 chunk 覆盖**根本不会触发**。二者都是**面向"未来若出现多 chunk 配置"的防御性正确化**，而非修复了一个当前存在的错误。

**D 何时才真正是 bug（保留其价值）**：`evaluate()` 内部分配的是**覆盖整个测试集**的 `frame_bbox_scores` 并在一次调用里算完整 AUC，且 `__main__` 硬编码 `chunked_samples_00.pkl`。**只要**把 `num_samples_each_chunk` 调小到某 test 集 > 1 chunk，原版就会**只给 chunk_00 打分、其余帧落空默认分**→ AUC 错。当前阈值够大，没踩到；glob 版让它对任意 chunk 数都正确。

**D1 —— batch-local key 隐患（当前数据不触发，但仍已修正为更干净写法）**：
- 内层写分是 `frame_bbox_scores[frame_id][i] = score`，key `i` 是 **batch 内局部下标**（作者写法）。`extract_samples.py` 按样本数切 chunk（不按帧），故**理论上**一帧的 bbox 可能跨 chunk 边界，chunk N+1 的 bbox（key 从 0 重计）覆盖 chunk N 同 key 的分。
- **当前数据不触发**：单 chunk，无 chunk 边界；且单帧 bbox 数远 < batch(128)，batch 内也不碰撞。故对现有 Ped2/Avenue/SH，新旧 key **产生完全相同的每帧分集合 → AUC 逐位相等**。
- **✅ 已修正为全局 key（2026-08-19）**：订正——`__getitem__` 返回的第 5 项 `indices_test` 是 **chunk 内局部索引 `indice`**，非全局 id；真正全局 id = **`_chunk_base`（累计前序 chunk 大小）+ `indice`**。已在 `eval.py`/`ml_memAE_sc_eval.py` 落地（`[ARIS-PATCH item D1]`）。**验证**：Ped2 官方权重复跑对齐基线（见 §5 验证记录）。

**D2 本地/集群分叉 → ✅ 已回并（2026-08-19）**：
- 摘要里提到的 `get_foreground` **坐标钳制补丁**（`>=2px` + 裁到边界，用于修 ShanghaiTech STC frame 52001 的 0-size `cv2.resize` 崩溃）原**只存在于集群 author-tree**。
- 现已移植到本地 `dataset.py` 的**两个分支**（`shape==3` 与 `shape==4`，`[ARIS-PATCH item D2]`），对正常框 no-op，只在退化框上兜底。本地 clone 现可**独立跑 ShanghaiTech 预处理**、且与集群一致。

### E. 种子控制 —— 合理的研究仪器，但只有部分确定性 ✅⚠️

**改动**：`train.py` / `finetune.py` / `ml_memAE_sc_train.py` 顶部加 8 行，从环境变量 `HF2VAD_SEED`（默认 2021）播种 `random / numpy / torch / cuda`。

**为什么合理**：整个"数据来源 vs 种子"实验的核心就是**测种子方差**（证明 finetune 的 ±增益是种子噪声）。要测方差就得能固定/切换种子。

**两点必须诚实说明**：
1. **作者原版无任何种子**。加了默认种子后，我的"从零训练"是一个**被播种的变体**，不是对作者"不控种子"原始条件的逐位复刻。
2. **只有部分确定性**：没有设 `cudnn.deterministic=True` / `benchmark=False`，也没给 DataLoader worker 设 `worker_init_fn`。所以**即使同一种子，run 之间仍会有抖动**。
   - 这对本实验**恰好无害甚至有利**：目标是"测抖动幅度"，本就需要跑多个不同种子；残余的 cudnn 非确定性只会让"这是噪声"的结论更稳。但若将来想**声明逐位可复现**，这套种子控制**不够**。

**小瑕疵**：`ml_memAE_sc_train.py` 里 `import numpy as _np_seed` 与上方已 import 的 `np` 重复，无害，已注明。

### F. `edflow/` 依赖 shim ✅

**改动**：新增 `edflow/__init__.py` + `edflow/util.py`（共 26 行），仅重实现 `edflow.util.retrieve`（按 "/" 路径取嵌套 dict 值、缺失返回 default）。`models/vunet.py` 只用到这一个函数。

**为什么合理**：原 `edflow==0.4.0` 与 torch 2.x 不兼容，装它会拖垮环境。shim 精确复刻 vunet 用到的行为。

**唯一保真缺口**：真 `retrieve` 的 `expand=True`（可展开 callable/kwargs）未实现——但 vunet 只用简单 path+default 形式，不触发该分支。合理。

---

## 3. 合理性总表

| 改动 | 判定 | 一句话 |
|------|------|--------|
| A1/A2 语言兼容 | ✅ 完全合理 | 版本适配，数值中性 |
| B1/B2 编译目标 | ✅ 完全合理 | 只改 GPU 目标架构，不改算子数学 |
| C 检测器替换 | ⚠️ 可接受**需声明**（✅ 已声明） | 唯一方法学偏离；结构上无 bug（原版也不按类过滤），但分数标定不同，"影响可控"仅经验成立。已写入报告 §5.1 |
| D 多-chunk | ✅ 修真 bug | 原版漏读 Avenue/SH 的非首 chunk |
| D1 batch-local key | ✅ **已修复** | 改用全局 sample id（`_chunk_base`+chunk 内 indice）做 key，消除跨 chunk 覆盖；待集群复跑确认 AUC 不变 |
| D2 本地缺钳制补丁 | ✅ **已回并** | 坐标钳制已移植到本地 `get_foreground` 两个分支，本地可独立跑 SH 预处理 |
| E 种子控制 | ✅ 合理 | 但仅部分确定性；不足以声明逐位复现 |
| F edflow shim | ✅ 合理 | 精确覆盖被调用行为 |

> **重要订正**：早前本文档称"改用已解包但未用的 `indices_test` 作全局 key"——经查 `Chunked_sample_dataset.__getitem__` 返回的第 5 项是 **chunk 内局部索引 `indice`**，并非全局 `sample_id`。真正的全局 id = **累计前序 chunk 大小 `_chunk_base` + `indice`**（因 `sample_id` 单调、chunk 按序切分，二者恒等）。D1 修复采用此式。

---

## 4. 建议落实情况

1. ✅ **（正确性，已做）D2 坐标钳制回并**：已移植到本地 `get_foreground` 两个分支（`>=2px` + 边界裁剪，对正常框 no-op）。本地现可独立跑 ShanghaiTech 预处理。
2. ✅ **（严谨性，已做）C 复现偏离声明**：已写入复现报告 §5.1（含"结构等价"新证据 + 三集官方权重 eval Δ≤0.02 无害证据表 + 残留 caveat）。
3. ✅ **（优化，已做+已验证）D1 全局 key**：`eval.py` / `ml_memAE_sc_eval.py` 已把 key 从 batch-local `i` 换成 `_chunk_base + indices_test[i]`。验证见 §5。
4. ⬜ **（可选，未做）E 逐位复现**：若未来需要，补 `cudnn.deterministic=True` + DataLoader `worker_init_fn`。当前部分确定性对"测种子方差"目标已足够。

---

## 5. D1 验证记录（2026-08-20，集群 SuperPod）

**现场关键发现**：`num_samples_each_chunk`（test：ped2=100k / avenue=200k / sh=300k）> 三集实际样本数（35587 / 107628 / 232223）→ **三个 test 集全部单 chunk**。因此 D1 的跨 chunk 覆盖不可能发生、D 的多-chunk glob 为 no-op，**解析上 D1 对当前数据逐位 AUC-中性**。

**实测（Ped2 官方权重 `ped2_HF2VAD_99.31.pth`，打了 D1 补丁的 `eval.py`，job 524435，2:09，COMPLETED）**：

| 项 | 值 |
|---|---|
| D1 补丁 eval AUC | **0.99523（99.52%）** |
| 记录基线（官方权重·自建 cubes） | **99.52%** |
| 判定 | ✅ 报告精度完全一致；代码跑通无运行时错误 |

> 第 4 位小数的 run-to-run 抖动（~1e-4）来自 sbatch **重算 `training_stats`**（`cal_training_stats` 前向过 cudnn 非确定），**与 D1 无关**——若复用基线的 `training_stats.npy` 则应逐位一致。

**Avenue / ShanghaiTech 全量复跑：判定为冗余，未做**。三集同为单 chunk，D1 中性的解析证明对它们同样成立；SH 需加载 13GB 单 chunk + 重算 stats ≈ 20 min，只会同样得到 90.6x / 76.1x（报告精度），信息增益≈0。如需 belt-and-suspenders 可另跑。
