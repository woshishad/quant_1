# 量化赛题模拟工程

这是一个面向 2026 量化交易研究大赛的本地模拟工程，用来在正式数据发布前先搭建数据生成、训练、验证和推理的完整链路。

## 项目内容

- 生成符合赛题格式的训练集和测试集，包含 `row_id`、`time_id`、`asset_id`、`feature_*`、`responder_*`、`target` 和 `weight`
- 让 `weight` 由少数特征主导，方便后续检查模型是否识别出关键特征
- 让 `target` 成为围绕 0 波动的连续回归目标，保留正负方向和幅度信息
- 训练一个带权重的线性融合模型，并导出可加载的模型包
- 通过 `main.py` 完成离线推理并生成提交文件

## 文件说明

- `generate_data.py`：生成模拟数据集
- `train.py`：训练 `target` 模型、`weight` 辅助模型，并导出分析结果
- `main.py`：加载模型包，读取测试集并输出提交文件
- `synthetic_competition/`：数据、指标、模型和分析工具

## 快速开始

```bash
conda env create -f environment.yml
conda activate quant-competition-sim
```

然后再运行：

```bash
python generate_data.py --output-dir data/synthetic
python train.py --data-dir data/synthetic --artifacts-dir artifacts
python main.py --data-dir data/synthetic --model-path artifacts/model_bundle.json --output-path submission.csv
```

## 说明

- 如果安装了 `pyarrow`，会优先写入 Parquet；否则自动回退到 CSV
- 训练时采用严格的时间顺序切分，避免未来信息泄露
- `weight` 既参与 `target` 的加权训练，也作为辅助回归目标，方便分析关键特征

## 外部方案调研

- [GitHub 相似比赛仓库调研与整合结果](github_similar_competitions_integration_20260825.md)：相似仓库、许可证、可迁移方法、已运行候选及冻结 holdout 结论
- [私榜策略文件与 Time-Series API 要求](private_strategy_timeseries_requirements_20260825.md)：`main.py`、`Model.predict()`、历史窗口、资源限制和上传前检查单
- [相似比赛、公开项目与方法索引](similar_projects_methods_index_20260825.md)：GitHub、雪球线索、灵均历史比赛资料和方法适配分级
