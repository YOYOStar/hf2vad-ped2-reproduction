# Ped2 异常分数曲线（定性结果）

由 `eval.py` → `utils/eval_utils.save_evaluation_curves` 生成（异常分数经 medfilt k=17 平滑，红色阴影=真值异常区间）。

- `official/`：**官方权重 `ped2_HF2VAD_99.31.pth`**，frame-level **AUC = 0.9952**
  - `auc.png`：整体 ROC（area=0.9952，曲线紧贴左上角）
  - `anomaly_curve_1..12.png`：12 个测试视频各自的异常分数曲线
- `selftrain/`：**本复现从零训练的 finetuned 模型**，frame-level **AUC = 0.9769**
  - 同结构 12 张曲线 + `auc.png`

**读图**：正常帧分数贴近 0、异常区间分数显著抬升，两者清晰分离即对应高 AUC。两套（官方/自训练）都呈现正确的异常响应；官方权重的分离更干净（对应更高 AUC）。

> 注：Ped2 共 12 个测试视频；部分视频全程正常（无红色区间），其曲线应整体维持低分。
