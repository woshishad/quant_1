# 最终提交模型卡

## 提交文件

```text
results/final_recommended_submission_xgb_residual/submission.zip
```

该提交已通过 CSV、ZIP、来源一致性和防泄露元数据审计。提交文件包含
`3,217,458` 行，字段为 `row_id,target`。

## 模型结构

主模型是 7 月 13 日生成的 regime composite，主要包括：

- neutralized 逐行模型；
- panel market-relative 模型；
- Market32 横截面聚合模型；
- Exact-Market 条件融合；
- LogisticRegression 市场状态分类与条件 Ridge 幅度头。

主模型公式的最后一层为：

```text
regime_prediction = base_conditional + 0.20 * soft_conditional_regime_prediction
```

新增的 XGBoost 模型只使用 `feature_*` 和两个缺失模式指标。它训练在更早的
train partition 0-6，在最近 calibration 区间上不重新训练，因此可作为时间外的
独立残差信号。

最终公式：

```text
final_prediction = regime_prediction - 0.1860647991 * xgb_prediction
```

## 参数选择和防泄露

- XGBoost 主训练区间：`time_id <= 699999`。
- 残差系数拟合区间：`868480 <= time_id <= 878479`。
- 冻结验证区间：`878480 <= time_id <= 888479`。
- official test 只用于生成预测，不参与系数选择。
- 残差系数限制在 `[-0.25, 0.0]`，实际拟合值为 `-0.1860647991`。

## 本地指标

| 指标 | 原 regime 模型 | 最终模型 |
|---|---:|---:|
| 前半段 Weighted Zero-Mean R2 | 0.0061667534 | 0.0062386056 |
| 后半段 Weighted Zero-Mean R2 | 0.0059195968 | 0.0060127207 |
| 全段 Weighted Zero-Mean R2 | 0.0060421867 | 0.0061247598 |

XGBoost 单独在该区间的分数仅为 `0.0000728169`。改进来自它对主模型残差的
反向修正，不代表 XGBoost 单模型优于主模型。

## 输出分布

- mean：`-0.0111228817`
- std：`0.0712350913`
- min：`-0.6391055502`
- max：`0.8020585699`
- null / non-finite：`0`

## 主要文件

- 最终提交：`results/final_recommended_submission_xgb_residual/submission.zip`
- 审计报告：`results/final_recommended_submission_xgb_residual/audit_report.json`
- 验证报告：`results/final_recommended_submission_xgb_residual/validation_report.json`
- 完整指标：`results/final_recommended_submission_xgb_residual/source_metrics.json`
- 分块分数：`results/final_recommended_submission_xgb_residual/block_scores.csv`
- 生成脚本：`code/experiments/asset_all_tcn/blend_regime_with_xgb_residual.py`
