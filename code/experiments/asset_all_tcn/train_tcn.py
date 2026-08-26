from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class ExperimentConfig:
    seed: int = 42
    train_end_time: int = 39_999
    valid_start_time: int = 40_000
    valid_end_time: int = 49_999
    sequence_len: int = 4
    epochs: int = 30
    early_stop_patience: int = 10
    batch_size: int = 4096
    learning_rate: float = 5e-5
    tcn_channels: int = 16
    tcn_dropout: float = 0.4
    asset_embedding_dim: int = 8
    weight_decay: float = 1e-4
    shrink_min: float = 0.0
    shrink_max: float = 1.2
    use_amp: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all-asset TCN and evaluate official weighted zero-mean R2.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/asset_all_time50000"))
    parser.add_argument(
        "--selected-features-file",
        type=Path,
        default=Path("results/asset01_ga_lgbm_tcn/selected_features.csv"),
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/asset_all_tcn"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/asset_all_tcn"))
    parser.add_argument("--resume-tcn-file", type=Path, default=None)
    parser.add_argument("--sequence-len", type=int, default=ExperimentConfig.sequence_len)
    parser.add_argument("--epochs", type=int, default=ExperimentConfig.epochs)
    parser.add_argument("--early-stop-patience", type=int, default=ExperimentConfig.early_stop_patience)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=ExperimentConfig.learning_rate)
    parser.add_argument("--tcn-channels", type=int, default=ExperimentConfig.tcn_channels)
    parser.add_argument("--tcn-dropout", type=float, default=ExperimentConfig.tcn_dropout)
    parser.add_argument("--asset-embedding-dim", type=int, default=ExperimentConfig.asset_embedding_dim)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="auto 优先使用 CUDA；cuda 会强制使用显卡，没检测到 GPU 就直接报错。",
    )
    parser.add_argument("--amp", action="store_true", help="在 CUDA 上启用混合精度，更充分利用 5060 Ti。")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    """统一处理设备选择，避免以为在用 GPU、实际却悄悄掉回 CPU。"""
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("你指定了 --device cuda，但当前 PyTorch 没有检测到 CUDA。")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col.startswith("feature_")]


def load_selected_feature_names(path: Path, available_columns: list[str]) -> list[str]:
    # 先复用 asset01 GA 找到的因子，验证这些因子是否能迁移到所有标的。
    selected_frame = pd.read_csv(path)
    if "feature_name" in selected_frame.columns:
        names = selected_frame["feature_name"].astype(str).tolist()
    elif "feature_index" in selected_frame.columns:
        indices = selected_frame["feature_index"].astype(int).tolist()
        names = [available_columns[index] for index in indices]
    else:
        raise ValueError(f"{path} must contain feature_name or feature_index column")

    missing = sorted(set(names) - set(available_columns))
    if missing:
        raise ValueError(f"selected features are not present in dataset: {missing[:10]}")
    return names


def weighted_zero_mean_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    # 官方最终分数的核心形式：所有样本按照原始 weight 加权，和“全部预测为 0”比较。
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    denominator = float(np.sum(weight * y_true * y_true))
    if denominator <= 0:
        return 0.0
    numerator = float(np.sum(weight * (y_true - y_pred) ** 2))
    return 1.0 - numerator / denominator


def score_by_asset(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    scores = {}
    for asset_id, asset_frame in frame.groupby("asset_id", sort=True):
        scores[str(int(asset_id))] = weighted_zero_mean_r2(
            asset_frame["target"].to_numpy(),
            asset_frame[prediction_column].to_numpy(),
            asset_frame["weight"].to_numpy(),
        )
    return scores


def optimal_shrink(y_true: np.ndarray, prediction: np.ndarray, weight: np.ndarray, min_value: float, max_value: float) -> float:
    # 对单一路预测，最优 shrink 有闭式解：argmin sum(w * (y - s*p)^2)。
    denominator = float(np.sum(weight * prediction * prediction))
    if denominator <= 1e-18:
        return 0.0
    shrink = float(np.sum(weight * y_true * prediction) / denominator)
    return min(max_value, max(min_value, shrink))


def standardize_features(values: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 均值和标准差只在训练时间段拟合，验证时间段只复用训练统计量，避免未来信息泄漏。
    mean = np.nanmean(values[train_mask], axis=0).astype(np.float32)
    scale = np.nanstd(values[train_mask], axis=0).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    values = (values - mean) / scale
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return values, mean, scale


class AllAssetSequenceDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        x_selected: np.ndarray,
        asset_to_index: dict[int, int],
        sequence_len: int,
        split: str,
        config: ExperimentConfig,
        train_weight_mean: float,
    ):
        self.sequence_len = sequence_len
        self.asset_to_index = asset_to_index
        self.train_weight_mean = train_weight_mean
        self.groups: dict[int, dict[str, np.ndarray]] = {}
        self.samples: list[tuple[int, int]] = []

        for asset_id in sorted(asset_to_index):
            # 每个 asset 单独滑窗，绝不跨 asset 拼历史。
            asset_positions = np.flatnonzero(frame["asset_id"].to_numpy(dtype=np.int64) == asset_id)
            asset_frame = frame.iloc[asset_positions].reset_index(drop=True)
            group = {
                "x": x_selected[asset_positions].astype(np.float32),
                "target": asset_frame["target"].to_numpy(dtype=np.float32),
                "weight": asset_frame["weight"].to_numpy(dtype=np.float32),
                "row_id": asset_frame["row_id"].to_numpy(dtype=np.int64),
                "time_id": asset_frame["time_id"].to_numpy(dtype=np.int64),
                "asset_id": asset_frame["asset_id"].to_numpy(dtype=np.int64),
                "asset_index": np.full(len(asset_frame), asset_to_index[asset_id], dtype=np.int64),
            }
            self.groups[asset_id] = group

            for pos, time_id in enumerate(group["time_id"]):
                if pos < sequence_len - 1:
                    continue
                start = pos + 1 - sequence_len
                expected_window = np.arange(time_id - sequence_len + 1, time_id + 1, dtype=np.int64)
                # asset 12 存在缺失时间点；含缺口的窗口不能当作连续时序样本。
                if not np.array_equal(group["time_id"][start : pos + 1], expected_window):
                    continue
                if split == "train" and time_id <= config.train_end_time:
                    self.samples.append((asset_id, pos))
                elif split == "valid" and config.valid_start_time <= time_id <= config.valid_end_time:
                    self.samples.append((asset_id, pos))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        asset_id, pos = self.samples[index]
        group = self.groups[asset_id]
        start = pos + 1 - self.sequence_len
        x = group["x"][start : pos + 1]
        raw_weight = float(group["weight"][pos])
        train_weight = raw_weight / self.train_weight_mean if self.train_weight_mean > 0 else raw_weight
        return {
            "x": torch.from_numpy(x),
            "target": torch.tensor(group["target"][pos], dtype=torch.float32),
            "weight": torch.tensor(train_weight, dtype=torch.float32),
            "raw_weight": torch.tensor(raw_weight, dtype=torch.float32),
            "row_id": torch.tensor(group["row_id"][pos], dtype=torch.int64),
            "time_id": torch.tensor(group["time_id"][pos], dtype=torch.int64),
            "asset_id": torch.tensor(group["asset_id"][pos], dtype=torch.int64),
            "asset_index": torch.tensor(group["asset_index"][pos], dtype=torch.int64),
        }


class PrecomputedSequenceDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]):
        # 全标的训练样本多，逐条在 __getitem__ 里切滑窗会很慢；这里把窗口一次性预计算成张量。
        self.tensors = {
            "x": torch.from_numpy(arrays["x"].astype(np.float32, copy=False)),
            "target": torch.from_numpy(arrays["target"].astype(np.float32, copy=False)),
            "weight": torch.from_numpy(arrays["weight"].astype(np.float32, copy=False)),
            "raw_weight": torch.from_numpy(arrays["raw_weight"].astype(np.float32, copy=False)),
            "row_id": torch.from_numpy(arrays["row_id"].astype(np.int64, copy=False)),
            "time_id": torch.from_numpy(arrays["time_id"].astype(np.int64, copy=False)),
            "asset_id": torch.from_numpy(arrays["asset_id"].astype(np.int64, copy=False)),
            "asset_index": torch.from_numpy(arrays["asset_index"].astype(np.int64, copy=False)),
        }

    def __len__(self) -> int:
        return int(self.tensors["target"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: tensor[index] for name, tensor in self.tensors.items()}


def build_sequence_arrays(
    frame: pd.DataFrame,
    x_selected: np.ndarray,
    asset_to_index: dict[int, int],
    sequence_len: int,
    split: str,
    config: ExperimentConfig,
    train_weight_mean: float,
) -> dict[str, np.ndarray]:
    # 一次性构造某个 split 的所有连续窗口。asset 12 有缺失点，所以用端点时间差过滤非连续窗口。
    x_parts = []
    target_parts = []
    raw_weight_parts = []
    row_id_parts = []
    time_id_parts = []
    asset_id_parts = []
    asset_index_parts = []

    frame_asset_id = frame["asset_id"].to_numpy(dtype=np.int64)
    for asset_id in sorted(asset_to_index):
        asset_positions = np.flatnonzero(frame_asset_id == asset_id)
        asset_frame = frame.iloc[asset_positions]
        asset_x = x_selected[asset_positions].astype(np.float32, copy=False)
        time_id = asset_frame["time_id"].to_numpy(dtype=np.int64)
        if len(time_id) < sequence_len:
            continue

        candidates = np.arange(sequence_len - 1, len(time_id), dtype=np.int64)
        starts = candidates + 1 - sequence_len
        is_continuous = (time_id[candidates] - time_id[starts]) == (sequence_len - 1)
        if split == "train":
            in_split = time_id[candidates] <= config.train_end_time
        elif split == "valid":
            in_split = (time_id[candidates] >= config.valid_start_time) & (time_id[candidates] <= config.valid_end_time)
        else:
            raise ValueError(f"unknown split: {split}")
        positions = candidates[is_continuous & in_split]
        if len(positions) == 0:
            continue

        # 这里仍然有一个按窗口的循环，但只发生一次；训练时不会再重复切片。
        x_parts.append(np.stack([asset_x[pos + 1 - sequence_len : pos + 1] for pos in positions]).astype(np.float32))
        target_parts.append(asset_frame["target"].to_numpy(dtype=np.float32)[positions])
        raw_weight_parts.append(asset_frame["weight"].to_numpy(dtype=np.float32)[positions])
        row_id_parts.append(asset_frame["row_id"].to_numpy(dtype=np.int64)[positions])
        time_id_parts.append(time_id[positions])
        asset_id_parts.append(asset_frame["asset_id"].to_numpy(dtype=np.int64)[positions])
        asset_index_parts.append(np.full(len(positions), asset_to_index[asset_id], dtype=np.int64))

    if not x_parts:
        raise ValueError(f"no sequence windows for split={split}")
    raw_weight = np.concatenate(raw_weight_parts).astype(np.float32)
    return {
        "x": np.concatenate(x_parts, axis=0),
        "target": np.concatenate(target_parts).astype(np.float32),
        "raw_weight": raw_weight,
        "weight": (raw_weight / train_weight_mean).astype(np.float32) if train_weight_mean > 0 else raw_weight,
        "row_id": np.concatenate(row_id_parts).astype(np.int64),
        "time_id": np.concatenate(time_id_parts).astype(np.int64),
        "asset_id": np.concatenate(asset_id_parts).astype(np.int64),
        "asset_index": np.concatenate(asset_index_parts).astype(np.int64),
    }


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
        return x + self.net(x)


class TCNRegressor(nn.Module):
    def __init__(self, input_dim: int, asset_count: int, channels: int, asset_embedding_dim: int, dropout: float):
        super().__init__()
        self.asset_embedding = nn.Embedding(asset_count, asset_embedding_dim)
        model_input_dim = input_dim + asset_embedding_dim
        self.input_projection = nn.Conv1d(model_input_dim, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            TCNBlock(channels, dilation=1, dropout=dropout),
            TCNBlock(channels, dilation=2, dropout=dropout),
            TCNBlock(channels, dilation=4, dropout=dropout),
            TCNBlock(channels, dilation=8, dropout=dropout),
        )
        self.head = nn.Linear(channels + asset_embedding_dim, 1)

    def forward(self, x: torch.Tensor, asset_index: torch.Tensor) -> torch.Tensor:
        asset_embed = self.asset_embedding(asset_index.long())
        repeated_embed = asset_embed.unsqueeze(1).expand(-1, x.shape[1], -1)
        x = torch.cat([x, repeated_embed], dim=-1).transpose(1, 2)
        features = self.blocks(self.input_projection(x))[:, :, -1]
        features = torch.cat([features, asset_embed], dim=-1)
        return self.head(features).squeeze(-1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> tuple[float, pd.DataFrame]:
    model.eval()
    outputs = {"row_id": [], "time_id": [], "asset_id": [], "target": [], "weight": [], "tcn_raw_prediction": []}
    amp_enabled = bool(use_amp and device.type == "cuda")
    with torch.no_grad():
        for batch in loader:
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred = model(batch["x"].to(device), batch["asset_index"].to(device))
            pred = pred.float().cpu().numpy()
            outputs["row_id"].append(batch["row_id"].cpu().numpy())
            outputs["time_id"].append(batch["time_id"].cpu().numpy())
            outputs["asset_id"].append(batch["asset_id"].cpu().numpy())
            outputs["target"].append(batch["target"].cpu().numpy())
            outputs["weight"].append(batch["raw_weight"].cpu().numpy())
            outputs["tcn_raw_prediction"].append(pred)
    frame = pd.DataFrame({key: np.concatenate(value) for key, value in outputs.items()})
    frame = frame.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)
    score = weighted_zero_mean_r2(
        frame["target"].to_numpy(),
        frame["tcn_raw_prediction"].to_numpy(),
        frame["weight"].to_numpy(),
    )
    return score, frame


def train_tcn(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
) -> tuple[nn.Module, dict, pd.DataFrame]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    initial_score, initial_pred = evaluate(model, valid_loader, device, config.use_amp)
    best_score = float(initial_score)
    best_state = copy.deepcopy(model.state_dict())
    best_pred = initial_pred.copy()
    history = [{"epoch": -1, "train_loss": None, "valid_weighted_r2": float(initial_score)}]
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
            asset_index = batch["asset_index"].to(device)
            optimizer.zero_grad(set_to_none=True)
            # CUDA + AMP 可以更好地利用 5060 Ti；CPU 或未开 AMP 时自动保持普通 FP32。
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred = model(x, asset_index)
                loss = torch.sum(weight * (pred - y) ** 2) / torch.clamp(torch.sum(weight), min=1e-6)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(y)
            total_rows += len(y)

        valid_score, valid_pred = evaluate(model, valid_loader, device, config.use_amp)
        train_loss = total_loss / max(total_rows, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_weighted_r2": float(valid_score)})
        print(f"TCN epoch {epoch + 1}/{config.epochs}: loss={train_loss:.6f}, valid={valid_score:.6f}")
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
    return model, {"score": best_score, "history": history}, best_pred


def save_plots(results_dir: Path, history: pd.DataFrame, predictions: pd.DataFrame, metrics: dict) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["valid_weighted_r2"], marker="o", linewidth=1)
    plt.xlabel("epoch")
    plt.ylabel("official weighted R2")
    plt.title("All-Asset TCN Validation Curve")
    plt.tight_layout()
    plt.savefig(results_dir / "valid_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(predictions["target"], predictions["prediction"], s=4, alpha=0.18)
    limit = max(float(np.nanmax(np.abs(predictions[["target", "prediction"]].to_numpy()))), 1e-6)
    plt.plot([-limit, limit], [-limit, limit], color="black", linewidth=1)
    plt.xlabel("target")
    plt.ylabel("prediction")
    plt.title(f"Target vs Prediction, score={metrics['shrink_score']:.4f}")
    plt.tight_layout()
    plt.savefig(results_dir / "target_vs_prediction.png", dpi=160)
    plt.close()

    by_asset = metrics["shrink_score_by_asset"]
    labels = sorted(by_asset, key=lambda value: int(value))
    values = [by_asset[label] for label in labels]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xlabel("asset_id")
    plt.ylabel("weighted zero-mean R2")
    plt.title("Validation Score by Asset")
    plt.tight_layout()
    plt.savefig(results_dir / "score_by_asset.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        sequence_len=args.sequence_len,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        tcn_channels=args.tcn_channels,
        tcn_dropout=args.tcn_dropout,
        asset_embedding_dim=args.asset_embedding_dim,
        use_amp=args.amp,
    )
    set_seed(config.seed)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    data_path = args.data_dir / "train.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"missing dataset file: {data_path}")

    preview_columns = pq.ParquetFile(data_path).schema_arrow.names
    selected_features = load_selected_feature_names(args.selected_features_file, preview_columns)
    read_columns = ["row_id", "time_id", "asset_id", "weight", "target"] + selected_features
    frame = pd.read_parquet(data_path, columns=read_columns)
    frame = frame.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)

    asset_ids = [int(asset_id) for asset_id in sorted(frame["asset_id"].unique().tolist())]
    asset_to_index = {asset_id: index for index, asset_id in enumerate(asset_ids)}
    time_values = frame["time_id"].to_numpy(dtype=np.int64)
    train_mask = time_values <= config.train_end_time
    x_values = frame[selected_features].to_numpy(dtype=np.float32, copy=True)
    x_all, mean, scale = standardize_features(x_values, train_mask)
    train_weight_mean = float(frame.loc[train_mask, "weight"].mean())

    train_arrays = build_sequence_arrays(frame, x_all, asset_to_index, config.sequence_len, "train", config, train_weight_mean)
    valid_arrays = build_sequence_arrays(frame, x_all, asset_to_index, config.sequence_len, "valid", config, train_weight_mean)
    train_dataset = PrecomputedSequenceDataset(train_arrays)
    valid_dataset = PrecomputedSequenceDataset(valid_arrays)
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise ValueError(f"empty sequence dataset: train={len(train_dataset)}, valid={len(valid_dataset)}")

    device = resolve_device(args.device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(
        f"Training all-asset TCN on device={device}, assets={len(asset_ids)}, "
        f"features={len(selected_features)}, train_sequences={len(train_dataset)}, "
        f"valid_sequences={len(valid_dataset)}, amp={config.use_amp}"
    )

    model = TCNRegressor(
        input_dim=len(selected_features),
        asset_count=len(asset_ids),
        channels=config.tcn_channels,
        asset_embedding_dim=config.asset_embedding_dim,
        dropout=config.tcn_dropout,
    )
    if args.resume_tcn_file is not None:
        checkpoint = torch.load(args.resume_tcn_file, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"Loaded TCN checkpoint from {args.resume_tcn_file}")

    model, tcn_metrics, predictions = train_tcn(model, train_loader, valid_loader, device, config)
    shrink = optimal_shrink(
        predictions["target"].to_numpy(),
        predictions["tcn_raw_prediction"].to_numpy(),
        predictions["weight"].to_numpy(),
        config.shrink_min,
        config.shrink_max,
    )
    predictions["prediction"] = shrink * predictions["tcn_raw_prediction"]
    predictions["error"] = predictions["prediction"] - predictions["target"]
    raw_score = weighted_zero_mean_r2(
        predictions["target"].to_numpy(),
        predictions["tcn_raw_prediction"].to_numpy(),
        predictions["weight"].to_numpy(),
    )
    shrink_score = weighted_zero_mean_r2(
        predictions["target"].to_numpy(),
        predictions["prediction"].to_numpy(),
        predictions["weight"].to_numpy(),
    )

    history = pd.DataFrame(tcn_metrics["history"])
    metrics = {
        "config": asdict(config),
        "device": str(device),
        "data_dir": str(args.data_dir),
        "selected_features_file": str(args.selected_features_file),
        "selected_feature_count": int(len(selected_features)),
        "asset_ids": asset_ids,
        "train_rows": int(train_mask.sum()),
        "valid_rows": int(((time_values >= config.valid_start_time) & (time_values <= config.valid_end_time)).sum()),
        "train_sequences": int(len(train_dataset)),
        "valid_sequences": int(len(valid_dataset)),
        "raw_tcn_score": float(raw_score),
        "raw_tcn_score_by_asset": score_by_asset(predictions.assign(prediction=predictions["tcn_raw_prediction"]), "prediction"),
        "shrink": float(shrink),
        "shrink_score": float(shrink_score),
        "shrink_score_by_asset": score_by_asset(predictions, "prediction"),
    }

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": len(selected_features),
            "asset_ids": asset_ids,
            "asset_to_index": asset_to_index,
            "config": asdict(config),
        },
        args.model_dir / "tcn.pt",
    )
    metadata = {
        "selected_features": selected_features,
        "feature_mean": mean.astype(float).tolist(),
        "feature_scale": scale.astype(float).tolist(),
        "metrics": metrics,
    }
    (args.model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    predictions.to_csv(args.results_dir / "validation_predictions.csv", index=False)
    history.to_csv(args.results_dir / "train_history.csv", index=False)
    pd.DataFrame({"feature_name": selected_features}).to_csv(args.results_dir / "selected_features.csv", index=False)
    (args.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_plots(args.results_dir, history, predictions, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
