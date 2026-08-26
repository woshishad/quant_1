from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader


def load_base_module():
    # 复用上一版实验里的数据校验、GA 特征选择、指标计算和 Dataset，避免两份代码逻辑漂移。
    base_path = Path(__file__).resolve().parents[1] / "asset01_ga_lstm_cnn" / "train_ga_lstm_cnn.py"
    spec = importlib.util.spec_from_file_location("asset01_ga_lstm_cnn_base", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import base experiment from {base_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()
ASSET_IDS = BASE.ASSET_IDS


@dataclass
class ExperimentConfig:
    # 本版本的思路：保留 GA/Ridge 的稳定性，引入 LightGBM 的表格非线性能力，再用 TCN 学局部时序形态。
    seed: int = 42
    train_end_time: int = 39_999
    valid_start_time: int = 40_000
    valid_end_time: int = 49_999
    min_features: int = 16
    max_features: int = 96
    ga_population: int = 40
    ga_generations: int = 20
    ga_crossover: float = 0.8
    ga_mutation: float = 0.03
    ga_elitism: int = 2
    ridge_alpha: float = 1.0
    sequence_len: int = 64
    epochs: int = 3
    early_stop_patience: int = 0
    batch_size: int = 1024
    learning_rate: float = 3e-4
    tcn_channels: int = 64
    tcn_dropout: float = 0.15
    asset_embedding_dim: int = 4
    lgbm_estimators: int = 400
    lgbm_learning_rate: float = 0.03
    lgbm_num_leaves: int = 31
    lgbm_min_child_samples: int = 800
    lgbm_subsample: float = 0.8
    lgbm_colsample_bytree: float = 0.8
    lgbm_reg_alpha: float = 0.0
    lgbm_reg_lambda: float = 10.0
    lgbm_objective: str = "regression"
    use_tabular_asset_feature: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GA + Ridge + LightGBM + TCN asset01 experiment.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/asset01_time50000"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/asset01_ga_lgbm_tcn"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/asset01_ga_lgbm_tcn"))
    parser.add_argument("--selected-features-file", type=Path, default=None)
    parser.add_argument("--ga-history-file", type=Path, default=None)
    parser.add_argument("--resume-tcn-file", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=ExperimentConfig.epochs)
    parser.add_argument("--early-stop-patience", type=int, default=ExperimentConfig.early_stop_patience)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=ExperimentConfig.learning_rate)
    parser.add_argument("--sequence-len", type=int, default=ExperimentConfig.sequence_len)
    parser.add_argument("--ridge-alpha", type=float, default=ExperimentConfig.ridge_alpha)
    parser.add_argument("--ga-generations", type=int, default=ExperimentConfig.ga_generations)
    parser.add_argument("--ga-population", type=int, default=ExperimentConfig.ga_population)
    parser.add_argument("--tcn-channels", type=int, default=ExperimentConfig.tcn_channels)
    parser.add_argument("--tcn-dropout", type=float, default=ExperimentConfig.tcn_dropout)
    parser.add_argument("--asset-embedding-dim", type=int, default=ExperimentConfig.asset_embedding_dim)
    parser.add_argument("--lgbm-estimators", type=int, default=ExperimentConfig.lgbm_estimators)
    parser.add_argument("--lgbm-learning-rate", type=float, default=ExperimentConfig.lgbm_learning_rate)
    parser.add_argument("--lgbm-num-leaves", type=int, default=ExperimentConfig.lgbm_num_leaves)
    parser.add_argument("--lgbm-min-child-samples", type=int, default=ExperimentConfig.lgbm_min_child_samples)
    parser.add_argument("--lgbm-subsample", type=float, default=ExperimentConfig.lgbm_subsample)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=ExperimentConfig.lgbm_colsample_bytree)
    parser.add_argument("--lgbm-reg-alpha", type=float, default=ExperimentConfig.lgbm_reg_alpha)
    parser.add_argument("--lgbm-reg-lambda", type=float, default=ExperimentConfig.lgbm_reg_lambda)
    parser.add_argument("--lgbm-objective", type=str, default=ExperimentConfig.lgbm_objective)
    parser.add_argument("--use-tabular-asset-feature", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_selected_feature_indices(path: Path, columns: list[str]) -> np.ndarray:
    # 调参阶段可以复用已经由 GA 选出的特征，避免每组参数都重新跑一遍遗传算法。
    selected_frame = pd.read_csv(path)
    if "feature_index" in selected_frame.columns:
        selected_indices = selected_frame["feature_index"].astype(int).to_numpy()
    elif "feature_name" in selected_frame.columns:
        index_by_name = {name: idx for idx, name in enumerate(columns)}
        selected_indices = selected_frame["feature_name"].map(index_by_name).astype(int).to_numpy()
    else:
        raise ValueError(f"{path} must contain feature_index or feature_name column")

    if len(selected_indices) == 0:
        raise ValueError(f"{path} does not contain any selected feature")
    if selected_indices.min() < 0 or selected_indices.max() >= len(columns):
        raise ValueError(f"{path} contains feature index outside 0..{len(columns) - 1}")
    return np.asarray(sorted(set(selected_indices.tolist())), dtype=np.int64)


def load_or_create_ga_history(path: Path | None, selected_count: int) -> pd.DataFrame:
    # 如果复用特征时没有传 GA 历史，就生成一行占位记录，保证后续保存和画图流程不分叉。
    if path is not None and path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(
        [
            {
                "generation": 0,
                "best_fitness": np.nan,
                "mean_fitness": np.nan,
                "selected_count": selected_count,
                "global_best_fitness": np.nan,
            }
        ]
    )


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * 2
        self.net = nn.Sequential(
            nn.ConstantPad1d((padding, 0), 0.0),
            nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ConstantPad1d((padding, 0), 0.0),
            nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差连接让 TCN 在低信噪比数据上更稳定，不会因为层数增加就明显退化。
        return x + self.net(x)


class TCNRegressor(nn.Module):
    def __init__(self, input_dim: int, channels: int, asset_embedding_dim: int, dropout: float):
        super().__init__()
        self.asset_embedding = nn.Embedding(len(ASSET_IDS), asset_embedding_dim)
        model_input_dim = input_dim + asset_embedding_dim
        self.input_projection = nn.Conv1d(model_input_dim, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            TCNBlock(channels, dilation=1, dropout=dropout),
            TCNBlock(channels, dilation=2, dropout=dropout),
            TCNBlock(channels, dilation=4, dropout=dropout),
            TCNBlock(channels, dilation=8, dropout=dropout),
        )
        self.head = nn.Linear(channels + asset_embedding_dim, 1)

    def forward(self, x: torch.Tensor, asset_id: torch.Tensor) -> torch.Tensor:
        # asset embedding 会在每个时间步重复拼接，让模型能学习不同标的的截距和动态差异。
        asset_embed = self.asset_embedding(asset_id.long())
        repeated_embed = asset_embed.unsqueeze(1).expand(-1, x.shape[1], -1)
        x = torch.cat([x, repeated_embed], dim=-1).transpose(1, 2)
        features = self.blocks(self.input_projection(x))[:, :, -1]
        features = torch.cat([features, asset_embed], dim=-1)
        return self.head(features).squeeze(-1)


def fit_ridge(
    x_selected: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    asset_id: np.ndarray,
    train_mask: np.ndarray,
    valid_sequences,
    config: ExperimentConfig,
) -> tuple[dict, pd.DataFrame, Ridge]:
    # Ridge 是低信噪比金融数据里的稳健基线，也作为融合中的保守预测源。
    sample_weight = BASE.balanced_sample_weight(weight[train_mask], asset_id[train_mask])
    model = Ridge(alpha=config.ridge_alpha, solver="lsqr", max_iter=500)
    model.fit(x_selected[train_mask], y[train_mask], sample_weight=sample_weight)
    pred_frame = predict_tabular_model(
        model,
        valid_sequences,
        "ridge",
        use_asset_feature=config.use_tabular_asset_feature,
    )
    score, by_asset = BASE.equal_asset_weighted_r2(
        pred_frame["target"].to_numpy(),
        pred_frame["prediction"].to_numpy(),
        pred_frame["weight"].to_numpy(),
        pred_frame["asset_id"].to_numpy(),
    )
    return {"score": float(score), "by_asset": by_asset}, pred_frame, model


def fit_lightgbm(
    x_selected: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    asset_id: np.ndarray,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    valid_sequences,
    config: ExperimentConfig,
) -> tuple[dict, pd.DataFrame, LGBMRegressor]:
    # LightGBM 适合表格因子：能捕捉非线性和特征交互，同时比深度模型更稳、更快。
    sample_weight = BASE.balanced_sample_weight(weight[train_mask], asset_id[train_mask])
    feature_names = [f"tabular_feature_{idx:03d}" for idx in range(x_selected.shape[1])]
    x_train = pd.DataFrame(x_selected[train_mask], columns=feature_names)
    x_valid = pd.DataFrame(x_selected[valid_mask], columns=feature_names)
    model = LGBMRegressor(
        objective=config.lgbm_objective,
        n_estimators=config.lgbm_estimators,
        learning_rate=config.lgbm_learning_rate,
        num_leaves=config.lgbm_num_leaves,
        min_child_samples=config.lgbm_min_child_samples,
        subsample=config.lgbm_subsample,
        subsample_freq=1,
        colsample_bytree=config.lgbm_colsample_bytree,
        reg_alpha=config.lgbm_reg_alpha,
        reg_lambda=config.lgbm_reg_lambda,
        random_state=config.seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        x_train,
        y[train_mask],
        sample_weight=sample_weight,
        eval_set=[(x_valid, y[valid_mask])],
        eval_sample_weight=[BASE.balanced_sample_weight(weight[valid_mask], asset_id[valid_mask])],
    )
    pred_frame = predict_tabular_model(
        model,
        valid_sequences,
        "lgbm",
        use_asset_feature=config.use_tabular_asset_feature,
        feature_names=feature_names,
    )
    score, by_asset = BASE.equal_asset_weighted_r2(
        pred_frame["target"].to_numpy(),
        pred_frame["prediction"].to_numpy(),
        pred_frame["weight"].to_numpy(),
        pred_frame["asset_id"].to_numpy(),
    )
    return {"score": float(score), "by_asset": by_asset}, pred_frame, model


def predict_tabular_model(
    model,
    valid_sequences,
    name: str,
    use_asset_feature: bool = False,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    # Ridge/LightGBM 不吃历史窗口，只使用序列最后一个时间点的 selected features。
    rows = []
    batch_x = []
    for asset, pos in valid_sequences.samples:
        group = valid_sequences.groups[asset]
        features = group["x"][pos]
        if use_asset_feature:
            # 表格模型额外使用一个中心化的 asset 特征；深度模型仍然走 embedding。
            asset_value = np.array([float(group["asset_id"][pos] == ASSET_IDS[1]) - 0.5], dtype=np.float32)
            features = np.concatenate([features, asset_value])
        batch_x.append(features)
        rows.append(
            {
                "row_id": int(group["row_id"][pos]),
                "time_id": int(group["time_id"][pos]),
                "asset_id": int(group["asset_id"][pos]),
                "target": float(group["target"][pos]),
                "weight": float(group["weight"][pos]),
            }
        )
    batch_array = np.asarray(batch_x, dtype=np.float32)
    predict_input = pd.DataFrame(batch_array, columns=feature_names) if feature_names is not None else batch_array
    predictions = model.predict(predict_input)
    for row, prediction in zip(rows, predictions):
        row["prediction"] = float(prediction)
    return pd.DataFrame(rows).sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)


def evaluate_tcn(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float], pd.DataFrame]:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["x"].to(device), batch["asset_id"].to(device)).cpu().numpy()
            for i in range(len(pred)):
                rows.append(
                    {
                        "row_id": int(batch["row_id"][i]),
                        "time_id": int(batch["time_id"][i]),
                        "asset_id": int(batch["asset_id"][i]),
                        "target": float(batch["target"][i]),
                        "weight": float(batch["raw_weight"][i]),
                        "prediction": float(pred[i]),
                    }
                )
    frame = pd.DataFrame(rows).sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)
    score, by_asset = BASE.equal_asset_weighted_r2(
        frame["target"].to_numpy(),
        frame["prediction"].to_numpy(),
        frame["weight"].to_numpy(),
        frame["asset_id"].to_numpy(),
    )
    return score, by_asset, frame


def train_tcn(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
) -> tuple[nn.Module, dict, pd.DataFrame]:
    # TCN 仍然保存验证分数最好的 epoch。金融数据过拟合很快，所以默认 epochs 不高。
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    # 先评估初始状态；这样从 checkpoint 续训时，如果后续没有提升，也能保留原 checkpoint。
    initial_score, initial_by_asset, initial_pred = evaluate_tcn(model, valid_loader, device)
    best_score = float(initial_score)
    best_pred = initial_pred.copy()
    history = [
        {
            "epoch": -1,
            "train_loss": None,
            "valid_equal_asset_weighted_r2": float(initial_score),
            "valid_asset_scores": initial_by_asset,
        }
    ]
    print(f"TCN initial: valid={initial_score:.6f}")
    epochs_without_improvement = 0
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["target"].to(device)
            weight = batch["weight"].to(device)
            asset_id = batch["asset_id"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, asset_id)
            loss = torch.sum(weight * (pred - y) ** 2) / torch.clamp(torch.sum(weight), min=1e-6)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            total_rows += len(y)
        valid_score, valid_by_asset, valid_pred = evaluate_tcn(model, valid_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_rows, 1),
                "valid_equal_asset_weighted_r2": float(valid_score),
                "valid_asset_scores": valid_by_asset,
            }
        )
        print(f"TCN epoch {epoch + 1}/{config.epochs}: loss={history[-1]['train_loss']:.6f}, valid={valid_score:.6f}")
        if valid_score > best_score:
            best_score = float(valid_score)
            best_state = copy.deepcopy(model.state_dict())
            best_pred = valid_pred.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if config.early_stop_patience > 0 and epochs_without_improvement >= config.early_stop_patience:
                print(f"TCN early stop at epoch {epoch + 1}, best_valid={best_score:.6f}")
                break
    model.load_state_dict(best_state)
    return model, {"score": best_score, "by_asset": history[int(np.argmax([h["valid_equal_asset_weighted_r2"] for h in history]))]["valid_asset_scores"], "history": history}, best_pred


def find_best_blend_with_shrink(
    ridge_pred: pd.DataFrame,
    lgbm_pred: pd.DataFrame,
    tcn_pred: pd.DataFrame,
) -> tuple[dict[str, float], float, dict[str, float], pd.DataFrame]:
    # 融合先搜三模型非负权重，再搜 shrink。shrink 用来压低预测幅度，金融低信噪比下很常见。
    merged = ridge_pred.rename(columns={"prediction": "ridge_prediction"}).merge(
        lgbm_pred[["row_id", "prediction"]].rename(columns={"prediction": "lgbm_prediction"}),
        on="row_id",
        how="inner",
    )
    merged = merged.merge(
        tcn_pred[["row_id", "prediction"]].rename(columns={"prediction": "tcn_prediction"}),
        on="row_id",
        how="inner",
    )
    columns = {
        "ridge": merged["ridge_prediction"].to_numpy(),
        "lgbm": merged["lgbm_prediction"].to_numpy(),
        "tcn": merged["tcn_prediction"].to_numpy(),
    }
    best_weights = {"ridge": 0.0, "lgbm": 0.0, "tcn": 0.0, "shrink": 1.0}
    best_score = -np.inf
    best_by_asset = {}
    for ridge_weight in np.linspace(0.0, 1.0, 21):
        for lgbm_weight in np.linspace(0.0, 1.0, 21):
            tcn_weight = 1.0 - ridge_weight - lgbm_weight
            if tcn_weight < -1e-12:
                continue
            base_prediction = (
                ridge_weight * columns["ridge"]
                + lgbm_weight * columns["lgbm"]
                + max(0.0, tcn_weight) * columns["tcn"]
            )
            for shrink in np.linspace(0.0, 1.2, 25):
                prediction = shrink * base_prediction
                score, by_asset = BASE.equal_asset_weighted_r2(
                    merged["target"].to_numpy(),
                    prediction,
                    merged["weight"].to_numpy(),
                    merged["asset_id"].to_numpy(),
                )
                if score > best_score:
                    best_score = float(score)
                    best_by_asset = by_asset
                    best_weights = {
                        "ridge": float(ridge_weight),
                        "lgbm": float(lgbm_weight),
                        "tcn": float(max(0.0, tcn_weight)),
                        "shrink": float(shrink),
                    }
    merged["prediction"] = best_weights["shrink"] * (
        best_weights["ridge"] * columns["ridge"]
        + best_weights["lgbm"] * columns["lgbm"]
        + best_weights["tcn"] * columns["tcn"]
    )
    merged["error"] = merged["prediction"] - merged["target"]
    return best_weights, best_score, best_by_asset, merged


def save_plots(results_dir: Path, ga_history: pd.DataFrame, predictions: pd.DataFrame, metrics: dict) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(ga_history["generation"], ga_history["best_fitness"], label="best")
    plt.plot(ga_history["generation"], ga_history["mean_fitness"], label="mean")
    plt.xlabel("generation")
    plt.ylabel("equal-asset weighted R2")
    plt.title("GA Fitness Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "ga_fitness_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(predictions["target"], predictions["prediction"], s=7, alpha=0.25)
    limit = max(float(np.nanmax(np.abs(predictions[["target", "prediction"]].to_numpy()))), 1e-6)
    plt.plot([-limit, limit], [-limit, limit], color="black", linewidth=1)
    plt.xlabel("target")
    plt.ylabel("prediction")
    plt.title(f"Target vs Prediction, score={metrics['blend']['score']:.4f}")
    plt.tight_layout()
    plt.savefig(results_dir / "target_vs_prediction.png", dpi=160)
    plt.close()

    asset_labels = ["0", "1"]
    x = np.arange(len(asset_labels))
    width = 0.2
    plt.figure(figsize=(8, 5))
    for offset, key in [(-1.5 * width, "ridge"), (-0.5 * width, "lgbm"), (0.5 * width, "tcn"), (1.5 * width, "blend")]:
        values = [metrics[key]["by_asset"][asset] for asset in asset_labels]
        plt.bar(x + offset, values, width=width, label=key)
    plt.xticks(x, asset_labels)
    plt.xlabel("asset_id")
    plt.ylabel("weighted zero-mean R2")
    plt.title("Validation Score by Asset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "score_by_asset.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sequence_len=args.sequence_len,
        ridge_alpha=args.ridge_alpha,
        ga_generations=args.ga_generations,
        ga_population=args.ga_population,
        tcn_channels=args.tcn_channels,
        tcn_dropout=args.tcn_dropout,
        asset_embedding_dim=args.asset_embedding_dim,
        lgbm_estimators=args.lgbm_estimators,
        lgbm_learning_rate=args.lgbm_learning_rate,
        lgbm_num_leaves=args.lgbm_num_leaves,
        lgbm_min_child_samples=args.lgbm_min_child_samples,
        lgbm_subsample=args.lgbm_subsample,
        lgbm_colsample_bytree=args.lgbm_colsample_bytree,
        lgbm_reg_alpha=args.lgbm_reg_alpha,
        lgbm_reg_lambda=args.lgbm_reg_lambda,
        lgbm_objective=args.lgbm_objective,
        use_tabular_asset_feature=args.use_tabular_asset_feature,
    )
    set_seed(config.seed)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    frame = BASE.load_and_validate_dataset(args.data_dir)
    columns = BASE.feature_columns(frame)
    time_values = frame["time_id"].to_numpy(dtype=np.int64)
    train_mask = time_values <= config.train_end_time
    valid_mask = (time_values >= config.valid_start_time) & (time_values <= config.valid_end_time)
    x_all, mean, scale = BASE.standardize_features(frame, columns, train_mask)
    y = frame["target"].to_numpy(dtype=np.float32)
    weight = frame["weight"].to_numpy(dtype=np.float32)
    asset_id = frame["asset_id"].to_numpy(dtype=np.int64)

    selector_config = BASE.ExperimentConfig(
        seed=config.seed,
        train_end_time=config.train_end_time,
        valid_start_time=config.valid_start_time,
        valid_end_time=config.valid_end_time,
        min_features=config.min_features,
        max_features=config.max_features,
        ga_population=config.ga_population,
        ga_generations=config.ga_generations,
        ga_crossover=config.ga_crossover,
        ga_mutation=config.ga_mutation,
        ga_elitism=config.ga_elitism,
        ridge_alpha=config.ridge_alpha,
        sequence_len=config.sequence_len,
    )
    if args.selected_features_file is not None:
        selected_indices = load_selected_feature_indices(args.selected_features_file, columns)
        ga_history = load_or_create_ga_history(args.ga_history_file, len(selected_indices))
        print(f"Using selected features from {args.selected_features_file}")
    else:
        selector = BASE.GeneticFeatureSelector(
            x_train=x_all[train_mask],
            y_train=y[train_mask],
            w_train=weight[train_mask],
            asset_train=asset_id[train_mask],
            x_valid=x_all[valid_mask],
            y_valid=y[valid_mask],
            w_valid=weight[valid_mask],
            asset_valid=asset_id[valid_mask],
            config=selector_config,
        )
        selected_mask, ga_history = selector.run()
        selected_indices = np.flatnonzero(selected_mask)
    selected_features = [columns[idx] for idx in selected_indices]
    x_selected = x_all[:, selected_indices]
    if config.use_tabular_asset_feature:
        # Ridge/LightGBM 使用当前行的 asset_id；TCN 继续使用独立的 asset embedding。
        asset_feature = (asset_id == ASSET_IDS[1]).astype(np.float32).reshape(-1, 1) - 0.5
        x_tabular = np.concatenate([x_selected, asset_feature], axis=1)
    else:
        x_tabular = x_selected

    dataset_config = BASE.ExperimentConfig(
        train_end_time=config.train_end_time,
        valid_start_time=config.valid_start_time,
        valid_end_time=config.valid_end_time,
        sequence_len=config.sequence_len,
    )
    train_dataset = BASE.TimeSeriesDataset(frame, x_selected, config.sequence_len, "train", dataset_config)
    valid_dataset = BASE.TimeSeriesDataset(frame, x_selected, config.sequence_len, "valid", dataset_config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device={device}, selected_features={len(selected_features)}")

    ridge_metrics, ridge_pred, ridge_model = fit_ridge(x_tabular, y, weight, asset_id, train_mask, valid_dataset, config)
    print(f"Ridge valid={ridge_metrics['score']:.6f}")
    lgbm_metrics, lgbm_pred, lgbm_model = fit_lightgbm(
        x_tabular, y, weight, asset_id, train_mask, valid_mask, valid_dataset, config
    )
    print(f"LightGBM valid={lgbm_metrics['score']:.6f}")
    tcn = TCNRegressor(
        input_dim=len(selected_features),
        channels=config.tcn_channels,
        asset_embedding_dim=config.asset_embedding_dim,
        dropout=config.tcn_dropout,
    )
    if args.resume_tcn_file is not None:
        # 只恢复模型权重，不恢复优化器；用于在已有最优 TCN 上继续小学习率微调。
        checkpoint = torch.load(args.resume_tcn_file, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        tcn.load_state_dict(state_dict)
        print(f"Loaded TCN checkpoint from {args.resume_tcn_file}")
    tcn, tcn_metrics, tcn_pred = train_tcn(tcn, train_loader, valid_loader, device, config)

    blend_weights, blend_score, blend_by_asset, predictions = find_best_blend_with_shrink(ridge_pred, lgbm_pred, tcn_pred)
    if "global_best_fitness" in ga_history.columns and ga_history["global_best_fitness"].notna().any():
        ga_best_fitness = float(ga_history["global_best_fitness"].max())
    else:
        ga_best_fitness = None

    metrics = {
        "config": asdict(config),
        "device": str(device),
        "selected_features_file": str(args.selected_features_file) if args.selected_features_file is not None else None,
        "resume_tcn_file": str(args.resume_tcn_file) if args.resume_tcn_file is not None else None,
        "selected_feature_count": int(len(selected_features)),
        "train_sequences": int(len(train_dataset)),
        "valid_sequences": int(len(valid_dataset)),
        "ga_best_fitness": ga_best_fitness,
        "ridge": ridge_metrics,
        "lgbm": lgbm_metrics,
        "tcn": {"score": float(tcn_metrics["score"]), "by_asset": tcn_metrics["by_asset"]},
        "blend": {"score": float(blend_score), "by_asset": blend_by_asset, "weights": blend_weights},
    }

    torch.save({"state_dict": tcn.state_dict(), "input_dim": len(selected_features), "config": asdict(config)}, args.model_dir / "tcn.pt")
    lgbm_model.booster_.save_model(str(args.model_dir / "lightgbm.txt"))
    metadata = {
        "data_dir": str(args.data_dir),
        "selected_features": selected_features,
        "selected_indices": selected_indices.astype(int).tolist(),
        "feature_mean": mean[selected_indices].astype(float).tolist(),
        "feature_scale": scale[selected_indices].astype(float).tolist(),
        "ridge_model": {
            "alpha": float(ridge_model.alpha),
            "coef": ridge_model.coef_.astype(float).tolist(),
            "intercept": float(ridge_model.intercept_),
        },
        "metrics": metrics,
    }
    (args.model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    predictions.to_csv(args.results_dir / "validation_predictions.csv", index=False)
    pd.DataFrame({"feature_index": selected_indices, "feature_name": selected_features}).to_csv(
        args.results_dir / "selected_features.csv", index=False
    )
    ga_history.to_csv(args.results_dir / "ga_history.csv", index=False)
    (args.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_plots(args.results_dir, ga_history, predictions, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
