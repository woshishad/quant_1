# 全标的最终训练、预测与融合流程

> 2026-08-04 更新：当前推荐提交已切换为
> `results/final_recommended_submission_xgb_residual/submission.zip`。它在下述 regime
> composite 上增加了时间外 XGBoost 残差修正，完整说明见
> `docs/final_submission_model_card.md`。本文后续章节保留此前实验流程作为历史记录。

本项目历史 Windows 环境是 `quant-competition-sim`；当前 WSL 重新实验使用
`quant-competition-wsl`。先按 [WSL 环境说明](wsl_environment.md) 创建并验证环境。

WSL 中推荐调用：

```bash
conda run -n quant-competition-wsl python -u code/experiments/asset_all_tcn/train_tcn.py --device cuda --amp
```

下面的 `D:\conda-envs\...` 命令保留为旧 Windows 复现记录；新实验请改用
`conda run -n quant-competition-wsl python ...`。

历史环境名称：`quant-competition-sim`
推荐直接调用环境里的 Python：

```powershell
D:\conda-envs\quant-competition-sim\python.exe
```

## 1. 确认 GPU

```powershell
D:\conda-envs\quant-competition-sim\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

当前已确认 PyTorch 能识别：`NVIDIA GeForce RTX 5060 Ti`。

## 2. 当前推荐提交

当前优先推荐“8-block 时间递增加权 shrink 版”提交：

```text
results/final_recommended_submission/submission.zip
```

这个目录是从实验目录 `results/blend_final_120k_global_shrink_recency_weighted_8blocks` 固化出来的最终提交包，里面还包含 `validation_report.json`、hash、候选排名和复现命令。
最终模型说明见：

```text
docs/final_submission_model_card.md
results/final_recommended_submission/MODEL_CARD.md
```

可视化总览页：

```text
results/final_recommended_submission/report.html
```
另外可以用审计脚本复查最终包：

```powershell
D:\conda-envs\quant-competition-sim\python.exe -u .\code\experiments\asset_all_tcn\audit_submission_package.py `
  --final-dir .\results\final_recommended_submission `
  --source-dir .\results\blend_final_120k_global_shrink_recency_weighted_8blocks `
  --sample-submission .\data\raw\public_release_20260630\public_release_20260630\data\sample_submission.csv
```

审计报告输出在：

```text
results/final_recommended_submission/audit_report.json
```

文件级 manifest 输出在：

```text
results/final_recommended_submission/MANIFEST.json
results/final_recommended_submission/MANIFEST.csv
```

校验 manifest：

```powershell
D:\conda-envs\quant-competition-sim\python.exe -u .\code\experiments\asset_all_tcn\verify_final_artifact_manifest.py `
  --manifest .\results\final_recommended_submission\MANIFEST.json
```

如果要把某个新的实验目录提升为最终提交包，使用打包脚本：

```powershell
D:\conda-envs\quant-competition-sim\python.exe -u .\code\experiments\asset_all_tcn\package_final_submission.py `
  --source-dir .\results\blend_final_120k_global_shrink_recency_weighted_8blocks `
  --final-dir .\results\final_recommended_submission `
  --candidate-ranking .\results\final_submission_candidate_ranking.csv `
  --sample-submission .\data\raw\public_release_20260630\public_release_20260630\data\sample_submission.csv
```

这版只使用 120k 主模型预测，不混入 240k，也不做逐 asset 参数。它把 calibration 按时间切成 8 个连续 block，并对越新的 holdout block 给越高权重来选择全局 shrink。

最终 shrink：

```text
1.36
```

选择它的原因：

- 完整 calibration 最优 shrink 是 `1.40`，但最新 block 的最优 shrink 明显下降，说明市场状态有变化；
- 8-block 递增加权选择 `1.36`，比 `1.40` 稍微保守，但完整 calibration 分数几乎不损失；
- 参数只有一个全局 shrink，比逐 asset 融合更不容易过拟合。

对照结果：

| 版本 | shrink / 策略 | full calibration | 说明 |
|---|---:|---:|---|
| 原始 120k | 1.00 | 0.002980 | 最保守 |
| 8-block 递增加权 | 1.36 | 0.003238 | 当前优先推荐 |
| 半段策略选择版 | 1.40 | 0.003240 | 略激进一点 |
| 4-block holdout 平均 | 1.51 | 0.003219 | 最新 block 风险更高 |
| 逐 asset 保守融合 | per asset | 0.003363 | full calibration 最高，但自由度更大 |

8-block 递增加权命令：

```powershell
D:\conda-envs\quant-competition-sim\python.exe -u .\code\experiments\asset_all_tcn\calibrate_global_shrink_stability.py `
  --calibration .\results\asset_all_final_best_protocol_lookback120k\calibration_predictions.csv `
  --test .\results\asset_all_final_best_protocol_lookback120k\final_test_predictions.csv `
  --results-dir .\results\blend_final_120k_global_shrink_recency_weighted_8blocks `
  --min-shrink 0.80 `
  --max-shrink 1.80 `
  --shrink-step 0.01 `
  --block-count 8 `
  --selection-rule holdout_recency_weighted_best
```

## 3. 半段策略选择版

半段策略选择版把 calibration 再按时间切成前半/后半：

- 前半段 `868480..878479`：拟合候选策略参数；
- 后半段 `878480..888479`：选择泛化更好的策略。

最终选中的策略是 `left_global_shrink`：只使用 120k 预测，再乘一个全局 shrink。用完整 calibration 重新拟合后，最终 shrink 是 `1.4`。

策略选择版命令：

```powershell
D:\conda-envs\quant-competition-sim\python.exe -u .\code\experiments\asset_all_tcn\select_blend_strategy_from_calibration.py `
  --left-calibration .\results\asset_all_final_best_protocol_lookback120k\calibration_predictions.csv `
  --right-calibration .\results\asset_all_final_best_protocol_lookback240k\calibration_predictions.csv `
  --left-test .\results\asset_all_final_best_protocol_lookback120k\final_test_predictions.csv `
  --right-test .\results\asset_all_final_best_protocol_lookback240k\final_test_predictions.csv `
  --left-name lookback120k `
  --right-name lookback240k `
  --results-dir .\results\blend_final_120k_240k_strategy_selected `
  --min-left-weight 0.80 `
  --max-left-weight 1.00 `
  --step 0.01 `
  --min-shrink 0.80 `
  --max-shrink 1.60 `
  --shrink-step 0.01
```

## 4. 120k 主模型

120k lookback 是当前最稳的单模型提交：

```powershell
D:\conda-envs\quant-competition-sim\python.exe -u .\code\experiments\asset_all_tcn\final_train_predict.py `
  --results-dir .\results\asset_all_final_best_protocol_lookback120k `
  --model-dir .\models\asset_all_final_best_protocol_lookback120k `
  --train-lookback-time-points 120000 `
  --cal-time-points 20000 `
  --top-k-candidates 64 128 `
  --lgbm-num-leaves-candidates 15 31 `
  --lgbm-seeds 11 42 73 `
  --per-asset-feature-mode fixed_ranking `
  --per-asset-top-k-candidates 16 32 64 `
  --per-asset-lgbm-num-leaves-candidates 7 15 `
  --per-asset-lgbm-seeds 42
```

输出：

- `results/asset_all_final_best_protocol_lookback120k/submission.csv`
- `results/asset_all_final_best_protocol_lookback120k/submission.zip`
- `results/asset_all_final_best_protocol_lookback120k/metrics.json`

## 5. 240k 辅助模型

240k lookback 单独 calibration 分不如 120k，但和 120k 的 test 预测相关性约 `0.779`，可以作为融合辅助。

```powershell
D:\conda-envs\quant-competition-sim\python.exe -u .\code\experiments\asset_all_tcn\final_train_predict.py `
  --results-dir .\results\asset_all_final_best_protocol_lookback240k `
  --model-dir .\models\asset_all_final_best_protocol_lookback240k `
  --train-lookback-time-points 240000 `
  --cal-time-points 20000 `
  --top-k-candidates 64 128 `
  --lgbm-num-leaves-candidates 15 31 `
  --lgbm-seeds 11 42 73 `
  --per-asset-feature-mode fixed_ranking `
  --per-asset-top-k-candidates 16 32 64 `
  --per-asset-lgbm-num-leaves-candidates 7 15 `
  --per-asset-lgbm-seeds 42
```

## 6. 逐 Asset 保守融合候选

这版 full calibration 分最高，但后半段验证略低于策略选择版。适合当冲分备选。

```powershell
D:\conda-envs\quant-competition-sim\python.exe -u .\code\experiments\asset_all_tcn\blend_predictions_from_calibration.py `
  --left-calibration .\results\asset_all_final_best_protocol_lookback120k\calibration_predictions.csv `
  --right-calibration .\results\asset_all_final_best_protocol_lookback240k\calibration_predictions.csv `
  --left-test .\results\asset_all_final_best_protocol_lookback120k\final_test_predictions.csv `
  --right-test .\results\asset_all_final_best_protocol_lookback240k\final_test_predictions.csv `
  --left-name lookback120k `
  --right-name lookback240k `
  --results-dir .\results\blend_final_120k_240k_per_asset_conservative `
  --blend-mode per_asset `
  --min-left-weight 0.80 `
  --max-left-weight 1.00 `
  --step 0.01 `
  --min-shrink 0.80 `
  --max-shrink 1.60 `
  --shrink-step 0.01
```

## 7. GPU TCN Smoke / 调参

TCN/PyTorch 训练会走 CUDA。建议显式加 `--device cuda`，这样没有 GPU 时会直接失败。

```powershell
D:\conda-envs\quant-competition-sim\python.exe -u .\code\experiments\asset_all_tcn\train_tcn.py `
  --results-dir .\results\asset_all_tcn_cuda_amp_smoke `
  --model-dir .\models\asset_all_tcn_cuda_amp_smoke `
  --sequence-len 4 `
  --epochs 1 `
  --early-stop-patience 1 `
  --batch-size 8192 `
  --learning-rate 0.00005 `
  --tcn-channels 16 `
  --tcn-dropout 0.4 `
  --asset-embedding-dim 8 `
  --device cuda `
  --amp
```

更长一点的 TCN 调参可以从下面开始：

```powershell
D:\conda-envs\quant-competition-sim\python.exe -u .\code\experiments\asset_all_tcn\train_tcn.py `
  --results-dir .\results\asset_all_tcn_cuda_amp_tune01 `
  --model-dir .\models\asset_all_tcn_cuda_amp_tune01 `
  --sequence-len 8 `
  --epochs 20 `
  --early-stop-patience 6 `
  --batch-size 8192 `
  --learning-rate 0.00003 `
  --tcn-channels 32 `
  --tcn-dropout 0.45 `
  --asset-embedding-dim 8 `
  --device cuda `
  --amp
```

注意：目前 TCN 验证分低于 Ridge/LightGBM 表格模型，短期更建议把它当作弱辅助模型，而不是主模型。
