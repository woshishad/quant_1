# GitHub 相似比赛仓库调研与整合结果

日期：2026-08-25
检索范围：GitHub 公开仓库
本地验证口径：Weighted Zero-Mean R2，严格时间前推与冻结系数审计

## 1. 结论

找到了相似度很高的公开项目，最接近的是 Kaggle 的 **Jane Street Real-Time Market Data Forecasting**。它与本赛题都使用匿名特征、时间和资产索引、样本权重、多组辅助 responder、零均值加权 R2，以及按时间顺序调用的推理接口。

截至检索时，GitHub 仓库搜索没有找到“2026量化交易研究大赛”的同名公开仓库。这只能说明公开仓库标题、描述和 README 检索结果为 0，不能排除私有仓库或未被索引的代码。

本次没有直接复制外部代码，而是完成了三项项目内整合：

1. 固化相似仓库、许可证和方法适用性，避免以后重复检索或误用无许可证代码。
2. 新增统一实验入口，把外部思路转换为本赛题可运行、只读训练集的候选。
3. 新增通用冻结审计器，任何新信号都必须叠加到当前四层模型后，通过隔离校准和最终 holdout 才能晋级。

实际验证了长窗口滚动特征和横截面排名两个候选。两者都没有通过外层审计，因此**没有并入最终模型，也没有读取 official test 生成预测**。当前本地最佳分数仍为 `0.0064196889`。

## 2. 相似仓库快照

下面的 star、fork 和更新时间是 2026-08-25 的 GitHub API 快照，后续会变化。

| 仓库 | 相似度与可借鉴内容 | 快照 | 许可证与处理方式 |
|---|---|---:|---|
| [evgeniavolkova/kagglejanestreet](https://github.com/evgeniavolkova/kagglejanestreet) | Jane Street 2024 完整方案；时间切分、16 个相关特征的市场均值、每资产 1000 时点滚动统计、GRU、多 responder 辅助监督、6 模型平均 | 250 star / 82 fork；commit `8598feb` | 未检测到许可证；只参考公开方法描述，不复制源码 |
| [Pony-Li/kaggle-jane-street-25](https://github.com/Pony-Li/kaggle-jane-street-25) | README 标注 321/3757 铜牌；卷积 GRU、合成在线测试、在线增量更新 | 5 star / 0 fork；commit `b197f8d` | 未检测到许可证；只作方法和接口对照 |
| [Zicheng-Xie/Zicheng_Xie-2025](https://github.com/Zicheng-Xie/Zicheng_Xie-2025-Kaggle-Jane-Street-Real-Time-Market-Data-Forecasting-Competition-) | README 自报 67/3757 银牌；GBDT+NN 集成、lag、Transformer/TabM、裁剪和指标感知缩放 | 1 star / 0 fork；MIT；主分支 `13b7246` | 只迁移静态集成、裁剪和 feature-only 历史特征；推理期 responder/lag 不适用于当前 API |
| [Billy1900/JS-Kaggle-2025](https://github.com/Billy1900/JS-Kaggle-2025) | README 自报 Top 1%；LGB/XGB/CatBoost、加权 R²、warm-up/adjust/online 三阶段 | 3 star / 2 fork；主分支 `07658b2`；未检测到许可证 | 只迁移离线阶段模拟和静态异构集成；不照搬需要真值的 online learning |
| [william-o-s/jane-street-real-time-market-data-forecasting](https://github.com/william-o-s/jane-street-real-time-market-data-forecasting) | responder 日滞后、Ridge/XGBoost/神经网络组合、官方接口示例 | 1 star / 1 fork；commit `76cb9e7` | MIT；本次没有复制代码 |
| [osyuksel/kaggle-optiver-2024](https://github.com/osyuksel/kaggle-optiver-2024) | Optiver Trading at the Close 第 15 名；横截面排名、市场均值、lag/diff/rolling、purged CV、LightGBM/XGBoost 集成 | 18 star / 11 fork；commit `d9360ab` | MIT；本次只迁移通用建模思路 |
| [MingjieWang0606/Kaggle-Jane-Street-AE-MLP-xgb-TOP1](https://github.com/MingjieWang0606/Kaggle-Jane-Street-AE-MLP-xgb-TOP1) | 较早一届 Jane Street 第一名公开实现；去噪自编码器、多个 response 联合监督、带 gap 的时间切分、MLP + XGBoost | 51 star / 15 fork；commit `c979e35` | MIT；任务是二分类交易决策，相似度低于 2024 回归赛题 |

## 3. 方法适配判断

| 外部方法 | 本赛题是否可用 | 当前整合状态 |
|---|---|---|
| 严格时间切分、purge gap、远期 gap 验证 | 可直接使用 | 项目已有；新审计器继续强制执行 |
| 同时刻市场均值、横截面 z-score 和 rank | 可直接使用，只依赖当前可见 feature | 已做独立候选并验证，未通过晋级门槛 |
| 每资产长窗口 rolling mean/std/deviation | 可直接使用，但必须严格 shift 后计算 | 已做 32/256/1000 时点候选并验证，未通过晋级门槛 |
| 多 responder 辅助监督 | 可用，但测试时只能使用 feature 预测出的 responder | 项目现有辅助堆叠已覆盖，并已进入当前候选链路 |
| GRU 按时间序列建模 | 原理可用 | 项目已有 TCN/时序实验；GRU 仍是后续独立候选，不因外部成绩直接采用 |
| 多 seed/多架构平均 | 可用 | 当前模型本身已是多分支融合；新架构必须先证明独立增量 |
| 前一日 responder lag | 当前接口不可用 | 外部 Jane Street 接口会返还 lags，本赛题 `predict(test)` 不返还 target/responder，禁止接入 |
| 在线增量学习 | 当前接口不可用 | 没有推理期真值，不能照搬在线更新 |
| 反匿名化资产或外部行情映射 | 不应使用 | 赛题资产与 feature 匿名，且存在规则与泛化风险 |

## 4. 已运行的整合实验

两个候选都叠加在当前四层模型 `0.0064196889` 之上。冻结协议为：

- 系数拟合：`868480..872479`
- 隔离校准：`873480..877479`
- 最终 holdout：`878480..888479`
- 两段之间各保留 1000 个 `time_id` purge gap
- holdout 再拆成 8 个连续细块，至少 5 个必须改善

| 候选 | 冻结系数 | 拟合段增量 | 隔离校准增量 | holdout 增量 | holdout 正块 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 32/256/1000 长窗口滚动 Ridge 残差 | `-0.5192701` | `+0.0010923` | `-0.0008499` | `-0.0003842` | `3/8` | 拒绝 |
| 横截面均值/z-score/demean/rank Ridge 残差 | `-0.7473397` | `+0.0001765` | `-0.0002015` | `-0.0000640` | `3/8` | 拒绝 |

长窗口候选在全 outer 汇总上表面增加 `+0.0002195`，但这个值由最早的系数拟合段贡献，隔离校准和最后 holdout 都下降，属于明显的时间不稳定，不能用全段汇总分掩盖。

横截面候选在后续区间的诊断最优系数约为 `-0.15`，而早段拟合得到 `-0.7473`，说明方向可能存在但幅度漂移严重。由于校准和 holdout 已经被查看，不能再用同一数据把系数改成 `-0.15` 后宣称通过；如继续研究，必须在更早的嵌套时间折中预先确定收缩规则，再用未参与选择的新区间验证。

## 5. 项目内新增入口

- `code/experiments/asset_all_tcn/run_github_inspired_experiment.py`：提供 `long_horizon_rolling` 和 `cross_sectional_rank` 两个固定实验配置，只生成 calibration 信号。
- `code/experiments/asset_all_tcn/audit_frozen_signal_candidate.py`：将候选叠加到当前验证基线，拟合一次系数后冻结，并输出分段、细块与晋级结果。
- `tests/test_github_inspired_integration.py`：检查所有外部启发配置都强制跳过 official test，并验证加权残差系数计算。

WSL 复现示例：

```bash
cd /mnt/e/量化大赛
/mnt/d/conda-envs/quant-competition-sim/python.exe -u code/experiments/asset_all_tcn/run_github_inspired_experiment.py --profile cross_sectional_rank --run-id github_inspired_cross_sectional_rank_repeat
```

命令应整行执行，不要在命令中间回车。输出中的 `status=candidate_rejected` 是正常、可保留的研究结论，不应继续生成 official test 提交。

## 6. 后续优先级

1. GRU 只作为独立残差信号测试，使用本赛题自己的连续 `time_id` 分块，不照搬 Jane Street 的 date/day 结构。
2. 去噪自编码器只在更早训练折学习表示，再对当前强模型残差做冻结验证；不得用 full outer 选择维度或融合权重。
3. 若研究横截面系数收缩，先在 `868480` 之前建立多折嵌套协议，预先冻结收缩强度，再评估后续区间。
4. 在没有线上 A/B 结果前，继续把当前四层模型称为“本地最佳候选”，不要表述为比赛最优。
