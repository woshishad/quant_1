# WSL 训练环境

## 旧 Windows 环境

历史环境名是 `quant-competition-sim`，Windows Python 路径曾写成
`D:\conda-envs\quant-competition-sim\python.exe`。原始配置只有 Python 3.11、NumPy、Pandas 和可选的 PyArrow；实际实验还使用了 SciPy、scikit-learn、LightGBM、XGBoost、CatBoost、PyTorch、Matplotlib 等包。

## 当前 WSL 环境

WSL 配置文件是 `configs/environment-wsl.yml`，环境名为 `quant-competition-wsl`。
它使用 Python 3.12、CUDA 13 系列 PyTorch/XGBoost，以及 pip 的 CatBoost GPU wheel。
安装脚本同时注册 `Python (quant-competition-wsl)` Jupyter 内核，可在 VS Code
或 JupyterLab 中直接选择。

在项目根目录执行：

```bash
bash scripts/setup_wsl_env.sh
```

检查 GPU：

```bash
/usr/lib/wsl/lib/nvidia-smi -L
conda run -n quant-competition-wsl python scripts/check_wsl_gpu.py
```

WSL 的 `nvidia-smi` 通常位于 `/usr/lib/wsl/lib`，不一定默认加入 Conda 的
`PATH`；检查脚本会自动补上这个目录。

训练脚本中的 `--device auto|cuda|cpu` 仍然有效。需要强制使用显卡时显式传
`--device cuda`；如果环境没有 CUDA，脚本会直接报错，不会悄悄退回 CPU。

LightGBM 仍按 CPU 使用，这是当前实验脚本的默认路径；PyTorch、XGBoost 和
CatBoost 分别通过 GPU smoke test 后再用于正式实验。

常用训练入口：

```bash
conda activate quant-competition-wsl

# TCN / PyTorch
python -u code/experiments/asset_all_tcn/train_tcn.py --device cuda --amp

# XGBoost residual
python -u code/experiments/asset_all_tcn/tune_residual_protocol.py \
  --residual-model xgboost --xgb-device cuda

# CatBoost residual
python -u code/experiments/asset_all_tcn/tune_residual_protocol.py \
  --residual-model catboost --catboost-task-type GPU
```

完整训练前先把 `--data-dir`、`--raw-data-dir` 和输出目录显式写出来，避免把
新的实验结果覆盖到历史 `results/` 目录。
