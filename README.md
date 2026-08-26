# 量化交易研究工程

本仓库保存量化交易研究代码、实验配置、测试和研究文档。大规模数据、训练模型和实验输出不进入 Git，需在本地数据盘或对象存储中单独管理。

## 目录

- `code/experiments/asset_all_tcn/`：主时序模型、融合、滚动验证和提交审计脚本
- `code/experiments/asset01_ga_lgbm_tcn/`、`code/experiments/asset01_ga_lstm_cnn/`：Asset 01 实验
- `code/baselines/synthetic_competition/`：本地模拟赛题基线
- `code/tools/`：数据和特征分析工具
- `configs/`：Conda 环境和依赖配置
- `docs/`：赛题说明、模型卡和研究报告
- `scripts/`：环境检查和安装脚本
- `tests/`：自动化测试
- `reproduced_Quant/`：独立的 XGBoost 策略仓库子模块

## 本地数据和输出

以下目录被 `.gitignore` 排除：`data/`、`models/`、`results/`、`artifacts/`、`catboost_info/`。运行实验前，需要把相应数据放在本地目录，并把输出写入这些目录。

## 环境

CPU 基线环境见 `configs/environment.yml`，WSL CUDA 环境见 `configs/environment-wsl.yml`。GPU 环境只用于训练，最终策略仍需按比赛的 CPU、无网络约束进行验证。

## 子模块

首次克隆后初始化外部策略仓库：

```bash
git clone --recurse-submodules <repository-url>
```

已有主仓库则运行：

```bash
git submodule update --init --recursive
```
