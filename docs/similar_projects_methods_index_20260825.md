# 相似比赛、公开项目与方法索引

日期：2026-08-25

本索引服务于当前匿名多标的时序预测赛题。资料按证据强度分为：

- **A：** 官方赛题/官方发布包/可复现代码与 README；
- **B：** 公开仓库的方案说明或比赛报道；
- **C：** 搜索摘要、雪球文章或宣传材料，只作研究线索。

外部方法不能因为榜单名次或文章标题就直接并入当前模型。所有新信号必须使用当前赛题自己的数据、严格时间前推、purge gap、冻结系数和未参与选择的 holdout 重新验证。

## 1. 与当前赛题最接近的项目

| 来源 | 链接 | 证据 | 可借鉴内容 | 当前适配结论 |
|---|---|---:|---|---|
| Jane Street 2024 solution | [evgeniavolkova/kagglejanestreet](https://github.com/evgeniavolkova/kagglejanestreet) | A/B | 时间序列 CV、市场均值、按 asset 长滚动统计、GRU、多 responder 辅助监督、多个 seed 平均 | 最接近；可迁移因果窗口和辅助头，但仓库 README 需要约 100GB RAM/12GB GPU，不能直接当私榜包 |
| Jane Street 2025 bronze | [Pony-Li/kaggle-jane-street-25](https://github.com/Pony-Li/kaggle-jane-street-25) | A/B | Conv-GRU、合成在线测试、在线推理流程 | 可用于接口和状态机对照；本赛题没有推理期真实 responder/target，禁止照搬在线学习 |
| Jane Street 2025 silver | [Zicheng-Xie/Zicheng_Xie-2025](https://github.com/Zicheng-Xie/Zicheng_Xie-2025-Kaggle-Jane-Street-Real-Time-Market-Data-Forecasting-Competition-) | B | README 自报 67/3757；GBDT+NN 集成、lag、Transformer/TabM、预测裁剪与指标感知缩放 | 方法线索价值高；当前 API 没有真实 lag/responder/weight，先只验证 feature-only 历史特征、轻量集成和裁剪，不能复制在线 lag 管线 |
| Jane Street 2025 Top 1% 线索 | [Billy1900/JS-Kaggle-2025](https://github.com/Billy1900/JS-Kaggle-2025) | B | README 自报 Top 1%；LGB/XGB/CatBoost 集成、加权 R²、warm-up/adjust/online 三阶段训练 | 可迁移静态异构集成和离线阶段模拟；没有推理期真值时不能照搬 online learning，也不能把自报名次当作当前赛题证据 |
| Jane Street real-time | [william-o-s/jane-street-real-time-market-data-forecasting](https://github.com/william-o-s/jane-street-real-time-market-data-forecasting) | A/B | Ridge/XGBoost/神经网络组合、responder 相关思路、低延迟管线 | MIT；只迁移公开方法，不复制代码到提交包 |
| Optiver Trading at the Close | [osyuksel/kaggle-optiver-2024](https://github.com/osyuksel/kaggle-optiver-2024) | A/B | 横截面统计、market mean、lag/diff/rolling、purged CV、LightGBM/XGBoost 集成 | 可做当前特征工程候选；本地已验证的横截面和长窗口候选未通过 holdout |
| Jane Street denoising/MLP/XGB | [MingjieWang0606/Kaggle-Jane-Street-AE-MLP-xgb-TOP1](https://github.com/MingjieWang0606/Kaggle-Jane-Street-AE-MLP-xgb-TOP1) | A/B | 去噪自编码器、多 response 联合监督、带 gap 的时间切分 | 可作为低维表示候选；当前不因外部成绩直接启用 |

## 2. 其他可复现时序/匿名金融比赛

| 比赛/仓库 | 链接 | 主要方法 | 适配判断 |
|---|---|---|---|
| Ubiquant Market Prediction top 3% | [TuozhenLiu/Ubiquant-Market-Prediction](https://github.com/TuozhenLiu/Ubiquant-Market-Prediction) | 匿名 `investment_id`、`time_id`、300 特征、缺失 asset 的回归问题 | 数据结构很接近；重点参考按 asset/time 对齐、缺失面板处理和 DVC 复现 |
| Ubiquant Top 1% 线索 | [pinouche/ubiquant-kaggle-competition](https://github.com/pinouche/ubiquant-kaggle-competition) | README 自报 Top 1%；匿名面板回归与特征/模型实验 | 适合补充 Ubiquant 的特征筛选和验证思路；名次是仓库自报，迁移前必须在本赛题冻结 holdout 复现 |
| Ubiquant project | [pacifikus/HFT](https://github.com/pacifikus/HFT) | PyTorch、DVC、MLflow、可配置训练/推理流水线 | 工程治理可借鉴；最终私榜不能依赖 MLflow/DVC 服务 |
| G-Research Crypto Forecasting | [wimwimyam/kaggle-g-research-crypto-forecasting](https://github.com/wimwimyam/kaggle-g-research-crypto-forecasting) | 加密资产时间序列预测与 online inference | 可参考按时间更新和稳健评估；市场、字段和指标不等同 |
| QRT Data Challenge 2022 | [PirashanthR/QRT-Data-Challenge-2022](https://github.com/PirashanthR/QRT-Data-Challenge-2022) | 用过去收益学习正交线性因子，Householder 参数化，余弦目标，严格跨股票测试 | 对当前赛题最有价值的是“低维稳定因子 + 约束优化”；不能直接使用其 250 日收益输入 |
| Optiver public repos | [beingamanforever/Optiver-Trading-at-the-close](https://github.com/beingamanforever/Optiver-Trading-at-the-close), [GrigoriiTarasov/Optiver-Trading-at-the-Close-retrain-monitoring](https://github.com/GrigoriiTarasov/Optiver-Trading-at-the-Close-retrain-monitoring) | LightGBM、特征选择、CatBoost、概念漂移监控、重训 | 可参考漂移监控和轻量模型；概念漂移只能在本地诊断，不能私榜期间用未来标签重训 |

2026-08-25 通过 GitHub API 和仓库页面复核了上述链接，列出的仓库均可访问；Jane Street 2025 银牌仓库当前主分支提交为 `13b7246`，Top 1% 线索仓库为 `07658b2`。GitHub API 搜索没有以“2026量化交易研究大赛”同名公开的可识别仓库；这不排除私有仓库、未索引仓库或赛队未公开代码。star、fork、commit 和许可证会变化，使用前应重新核对。

## 3. 雪球检索结果

雪球直接访问受到 WAF/登录限制，本次没有把页面正文当作已核验事实。以下链接来自搜索索引，只保留为线索：

| 线索 | 链接 | 可见主题 | 证据级别 | 处理 |
|---|---|---|---:|---|
| CTA 方法 | [雪球文章](https://xueqiu.com/4382174960/289882545) | 趋势跟踪、统计套利等 CTA 分类 | C | 只能抽取通用研究关键词，不能采信宣传收益 |
| 量化 AI 模型 | [雪球文章](https://xueqiu.com/1985440678/305333936) | 量化模型、策略和工具讨论 | C | 需登录后人工核验正文、数据和回测区间 |
| 西蒙斯案例 | [雪球文章](https://xueqiu.com/8218662479/328876090) | 数学模型、统计建模的案例介绍 | C | 只能作为模型思想阅读材料 |
| 量化 AI/策略 | [雪球文章](https://xueqiu.com/1199144979/253238925) | 量化 AI 模型与策略 | C | 不能作为当前赛题分数证据 |

雪球值得继续搜的关键词：`Jane Street responder`、`Ubiquant time_id`、`匿名特征 回归`、`purged walk-forward`、`滚动窗口 量化`、`LightGBM 横截面`、`responder 辅助任务`。任何进入项目的结论都要回到当前数据复现。

## 4. 灵均相关比赛资料

目前能从灵均官网核验到的是历史的 **第二届中国（横琴）国际高校量化金融大赛**，不是当前匿名 feature/time_id 题的同一比赛：

- [灵均官网新闻：启动仪式](https://www.lingjuninvest.com/news/4470)：2019-04-29；说明北京大学深圳研究院、清华大学深圳研究生院等主办/承办单位及大赛启动背景。
- [搜狐：报名启动与章程](https://www.sohu.com/a/311703476_100000325)：策略开发、策略信息提交、样本外实盘模拟、策略报告和答辩；评审同时考虑收益、风险、逻辑稳健性和创新性。
- [广东石油化工学院获奖报道](https://site.gdupt.edu.cn/lxy/info/1134/2882.htm)：2020-12-25；记录第二届比赛由初赛、复赛和全国总决赛构成，复赛持续约七个月，依据回测、模拟交易和策略报告评审。
- [网易转述](https://www.163.com/news/article/EFKDB4OM000189DG.html)：说明优矿平台参与策略研发和比赛平台对接。

从这些材料能提取的通用做法是：样本外模拟、风险指标、策略报告、接口预演和答辩解释；不能提取出一个可直接复用的“灵均冠军模型”。公开资料没有给出可验证的特征、参数和源代码。

## 5. 方法优先级

### P0：当前最值得做

1. 将当前四层候选实现为实时 `Model.predict()`，历史窗口先固定为 32 个 `time_id`。
2. 保留当前 48 个稳定 feature 的共享 Ridge/轻量树模型作为底座。
3. 加入严格 shift 的 per-asset lag/delta/rolling 和有限市场历史；只在预测后更新缓存。
4. 对 weight/responder 做 feature-only 预测，作为辅助残差信号，不读取测试期真实 responder。
5. 使用冻结 gamma、purge gap、最后 holdout 和 8 个时间块审计，超过当前 `0.0064196889` 才允许晋级。

### P1：值得独立验证

- 轻量 GRU/TCN 只读取 32 步压缩窗口，蒸馏到 Ridge/小 MLP 后再测 CPU runner；
- 去噪自编码器或正交低维因子，只在早期折学习表示，再冻结到后续 holdout；
- 多 seed/多模型平均，但先测误差相关性和最差时间块，不因平均通常有效就直接加入；
- 概念漂移诊断、滚动校准和缩放参数固定，不能使用评估期未来标签。

### 不应照搬

- Jane Street 公开方案里的 online learning：当前 API 不提供推理期真实 target/responder；
- 需要 100GB RAM、GPU 或外部服务的模型；
- 前一日 responder lag，除非本赛题官方明确提供；
- 外部行情映射、资产反匿名化、未来测试数据预读；
- 雪球文章中的收益率、回撤和“年化”宣传数字，除非有完整可审计数据。

## 6. 统一晋级标准

任何外部方法进入正式候选前，必须同时满足：

- 严格时间前推；所有选择在训练折内部完成；
- 至少一个未参与调参的冻结 holdout 增量为正；
- 多数时间块改善，不能靠单一 asset 或单一早期区间贡献；
- `predict()` 不读取禁用字段，分块边界改变时结果一致；
- 4 核 CPU、12 GB RAM、本地官方 runner 无异常、无超时、无 NaN/Inf；
- 保存方法、数据范围、参数、依赖、hash 和失败原因。

## 7. 当前项目内已完成的整合

现有 [GitHub 相似比赛调研与整合](github_similar_competitions_integration_20260825.md) 已把长窗口滚动和横截面 rank 两个候选接入冻结审计；两者均未通过外层 holdout，因此没有并入最终模型，也没有继续生成 official test 提交。当前本地最佳仍应称为“本地候选”，不能称为线上最优。
