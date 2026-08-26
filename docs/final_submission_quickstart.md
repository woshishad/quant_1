# 最终提交快速指南

## WSL 环境

当前重新实验使用 WSL Conda 环境 `quant-competition-wsl`。首次准备环境：

```bash
bash scripts/setup_wsl_env.sh
conda run -n quant-competition-wsl python scripts/check_wsl_gpu.py
```

在 WSL 中运行 Python 脚本时，从项目根目录调用：

```bash
conda run -n quant-competition-wsl python -u code/experiments/asset_all_tcn/blend_regime_with_xgb_residual.py --results-dir results/blend_latest_regime_xgb_residual
```

下面的 `D:\conda-envs\...` 命令是旧 Windows 环境的历史复现命令；新实验请使用上面的 WSL 命令。

## 当前推荐提交

直接上传：

```text
results/final_recommended_submission_xgb_residual/submission.zip
```

当前方案以 7 月 13 日的 regime composite 为主预测，并加入一个训练区间更早的
XGBoost 反向残差修正：

```text
final_prediction = regime_prediction - 0.1860647991 * xgb_prediction
```

残差系数只使用 calibration 前半段 `868480..878479` 拟合，然后冻结到后半段
`878480..888479` 和 official test。

## 验证结果

| 指标 | 原 regime 模型 | XGB 残差修正版 | 提升 |
|---|---:|---:|---:|
| calibration 前半段 | 0.0061667534 | 0.0062386056 | +0.0000718522 |
| calibration 后半段 | 0.0059195968 | 0.0060127207 | +0.0000931239 |
| calibration 全段 | 0.0060421867 | 0.0061247598 | +0.0000825731 |

提交检查：

- rows：`3,217,458`
- columns：`row_id,target`
- row_id 顺序：与 sample submission 完全一致
- NaN/Inf：`0`
- ZIP 审计：`overall_passed=true`
- ZIP sha256：`929b6f75c7f57e1dd3b87a3f28f49239637d74f3aba74855da75c43f1ecae1bd`

完整审计文件：

```text
results/final_recommended_submission_xgb_residual/audit_report.json
results/final_recommended_submission_xgb_residual/validation_report.json
results/final_recommended_submission_xgb_residual/source_metrics.json
```

## 重新生成

在项目根目录运行，命令中间不要回车：

```powershell
D:\conda-envs\quant-competition-sim\python.exe -u .\code\experiments\asset_all_tcn\blend_regime_with_xgb_residual.py --results-dir .\results\blend_latest_regime_xgb_residual
```

重新固化并审计最终目录：

```powershell
D:\conda-envs\quant-competition-sim\python.exe -u .\code\experiments\asset_all_tcn\package_final_submission.py --source-dir .\results\blend_latest_regime_xgb_residual --final-dir .\results\final_recommended_submission_xgb_residual --candidate-ranking .\results\blend_latest_regime_xgb_residual\candidate_ranking.csv
```

## 历史版本

旧的 `results/final_recommended_submission/submission.zip` 和
`results/final_latest_regime_classification_model/submission.zip` 均保留，未被覆盖。
