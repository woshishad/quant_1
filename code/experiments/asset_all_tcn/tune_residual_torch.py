from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, Dataset

from final_train_predict import (
    BASE_COLUMNS_TRAIN,
    load_feature_ranking,
    parquet_paths,
    read_partitioned_frame,
    schema_columns,
    time_range,
)
from walk_forward_tabular import (
    apply_shrink,
    calibrate_shrink_info,
    score_candidate_on_calibration,
    standardize,
    summarize_shrink_info,
    weighted_zero_mean_r2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ridge 先预测 target，再用 PyTorch 深度模型在 GPU 上预测 Ridge 残差。"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw/public_release_20260630/public_release_20260630/data"),
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument(
        "--fixed-features-file",
        type=Path,
        default=Path("results/asset_all_stable_features_100k/selected_features_stable_top128.csv"),
    )
    parser.add_argument("--max-train-time-id", type=int, default=None)
    parser.add_argument("--lookback-time-points", type=int, default=90_000)
    parser.add_argument("--cal-time-points", type=int, default=20_000)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)

    parser.add_argument("--model", choices=["mlp", "tcn"], default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--asset-embedding-dim", type=int, default=8)
    parser.add_argument("--sequence-len", type=int, default=32)
    parser.add_argument("--tcn-channels", type=int, default=64)
    parser.add_argument("--tcn-levels", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="0 表示使用全部训练样本；TCN 可先设 200000/300000 控制耗时。",
    )

    parser.add_argument("--residual-weight-min", type=float, default=0.0)
    parser.add_argument("--residual-weight-max", type=float, default=1.0)
    parser.add_argument("--residual-weight-step", type=float, default=0.05)
    parser.add_argument("--shrink-mode", choices=["global", "per_asset"], default="per_asset")
    parser.add_argument("--shrink-cap-candidates", type=float, nargs="+", default=[1.0, 1.2, 1.4])
    parser.add_argument("--candidate-score-mode", choices=["full", "mean_halves", "min_halves"], default="min_halves")
    parser.add_argument("--save-validation-predictions", action="store_true")
    return parser.parse_args()


def json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def make_results_dir(path: Path | None, model_name: str) -> Path:
    if path is not None:
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"asset_all_residual_{model_name}_{timestamp}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda 已指定，但当前 PyTorch 没检测到 CUDA。")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_time_blocks(
    y_true: np.ndarray,
    prediction: np.ndarray,
    weight: np.ndarray,
    time_id: np.ndarray,
    block_count: int,
) -> dict[str, float]:
    unique_times = np.unique(time_id)
    chunks = [chunk for chunk in np.array_split(unique_times, block_count) if len(chunk) > 0]
    scores = []
    for chunk in chunks:
        mask = (time_id >= int(chunk[0])) & (time_id <= int(chunk[-1]))
        scores.append(float(weighted_zero_mean_r2(y_true[mask], prediction[mask], weight[mask])))
    values = np.asarray(scores, dtype=np.float64)
    return {
        f"block{block_count}_mean_score": float(np.mean(values)),
        f"block{block_count}_min_score": float(np.min(values)),
        f"block{block_count}_last_score": float(values[-1]),
        f"block{block_count}_negative_count": int(np.sum(values < 0.0)),
    }


def fit_ridge_baseline(
    fit_x: np.ndarray,
    cal_x: np.ndarray,
    y_fit: np.ndarray,
    w_fit: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, Ridge]:
    sample_weight = w_fit / max(float(np.mean(w_fit)), 1e-12)
    model = Ridge(alpha=float(alpha), solver="lsqr", max_iter=500)
    model.fit(fit_x, y_fit, sample_weight=sample_weight)
    return model.predict(fit_x), model.predict(cal_x), model


def load_window_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, dict, list[str], np.ndarray, np.ndarray]:
    train_paths = parquet_paths(args.raw_data_dir, "train")
    train_min_time, train_max_time_available = time_range(train_paths)
    train_end_time = (
        min(train_max_time_available, int(args.max_train_time_id))
        if args.max_train_time_id is not None
        else train_max_time_available
    )
    train_start_time = max(int(train_min_time), int(train_end_time) - int(args.lookback_time_points) + 1)
    fit_end_time = int(train_end_time) - int(args.cal_time_points)
    cal_start_time = fit_end_time + 1
    if fit_end_time < train_start_time:
        raise ValueError("lookback-time-points 太短，无法容纳 calibration 段。")

    available_columns = schema_columns(train_paths)
    ranking = load_feature_ranking(args.fixed_features_file, available_columns)
    selected_features = ranking.head(int(args.top_k))["feature_name"].astype(str).tolist()
    frame = read_partitioned_frame(
        train_paths,
        BASE_COLUMNS_TRAIN + selected_features,
        min_time=train_start_time,
        max_time=train_end_time,
    )
    time_values = frame["time_id"].to_numpy(dtype=np.int64)
    fit_mask = (time_values >= train_start_time) & (time_values <= fit_end_time)
    cal_mask = (time_values >= cal_start_time) & (time_values <= train_end_time)
    window_info = {
        "raw_train_min_time": int(train_min_time),
        "raw_train_max_time": int(train_max_time_available),
        "train_start_time": int(train_start_time),
        "fit_train_end_time": int(fit_end_time),
        "cal_start_time": int(cal_start_time),
        "train_end_time": int(train_end_time),
        "lookback_time_points": int(args.lookback_time_points),
        "fit_rows": int(fit_mask.sum()),
        "cal_rows": int(cal_mask.sum()),
    }
    return frame, window_info, selected_features, fit_mask, cal_mask


class TabularResidualDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        residual: np.ndarray,
        weight: np.ndarray,
        asset_index: np.ndarray,
        row_index: np.ndarray,
    ):
        self.x = x.astype(np.float32, copy=False)
        self.residual = residual.astype(np.float32, copy=False)
        self.weight = weight.astype(np.float32, copy=False)
        self.asset_index = asset_index.astype(np.int64, copy=False)
        self.row_index = row_index.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(len(self.residual))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[index]),
            "asset_index": torch.tensor(self.asset_index[index], dtype=torch.long),
            "target": torch.tensor(self.residual[index], dtype=torch.float32),
            "weight": torch.tensor(self.weight[index], dtype=torch.float32),
            "row_index": torch.tensor(self.row_index[index], dtype=torch.long),
        }


class SequenceResidualDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        x_all: np.ndarray,
        residual_all: np.ndarray,
        weight_all: np.ndarray,
        asset_to_index: dict[int, int],
        sequence_len: int,
        split_mask: np.ndarray,
        max_samples: int,
        seed: int,
    ):
        self.x_all = x_all.astype(np.float32, copy=False)
        self.residual_all = residual_all.astype(np.float32, copy=False)
        self.weight_all = weight_all.astype(np.float32, copy=False)
        self.asset_index_all = frame["asset_id"].map(asset_to_index).to_numpy(dtype=np.int64)
        self.sequence_len = int(sequence_len)
        self.samples: list[int] = []

        time_values = frame["time_id"].to_numpy(dtype=np.int64)
        asset_values = frame["asset_id"].to_numpy(dtype=np.int64)
        split_positions = np.flatnonzero(split_mask)
        split_position_set = set(int(pos) for pos in split_positions)

        for asset_id in sorted(asset_to_index):
            asset_positions = np.flatnonzero(asset_values == asset_id)
            asset_times = time_values[asset_positions]
            for local_pos in range(self.sequence_len - 1, len(asset_positions)):
                global_pos = int(asset_positions[local_pos])
                if global_pos not in split_position_set:
                    continue
                start_local = local_pos + 1 - self.sequence_len
                window_times = asset_times[start_local : local_pos + 1]
                # 严格连续的同一 asset 窗口才允许进入 TCN，避免缺失 time_id 造成伪序列。
                if window_times[-1] - window_times[0] == self.sequence_len - 1 and np.all(np.diff(window_times) == 1):
                    self.samples.append(global_pos)

        if max_samples and max_samples > 0 and len(self.samples) > max_samples:
            rng = np.random.default_rng(seed)
            chosen = rng.choice(np.asarray(self.samples, dtype=np.int64), size=int(max_samples), replace=False)
            self.samples = sorted(int(pos) for pos in chosen)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pos = self.samples[index]
        start = pos + 1 - self.sequence_len
        return {
            "x": torch.from_numpy(self.x_all[start : pos + 1]),
            "asset_index": torch.tensor(self.asset_index_all[pos], dtype=torch.long),
            "target": torch.tensor(self.residual_all[pos], dtype=torch.float32),
            "weight": torch.tensor(self.weight_all[pos], dtype=torch.float32),
            "row_index": torch.tensor(pos, dtype=torch.long),
        }


class MLPResidualNet(nn.Module):
    def __init__(self, input_dim: int, asset_count: int, asset_embedding_dim: int, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.asset_embedding = nn.Embedding(asset_count, asset_embedding_dim)
        blocks = []
        dim = input_dim + asset_embedding_dim
        for _ in range(int(layers)):
            blocks.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            dim = hidden_dim
        blocks.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, asset_index: torch.Tensor) -> torch.Tensor:
        embedding = self.asset_embedding(asset_index)
        return self.net(torch.cat([x, embedding], dim=1)).squeeze(-1)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + self.downsample(x))


class TCNResidualNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        asset_count: int,
        asset_embedding_dim: int,
        channels: int,
        levels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        blocks = []
        in_channels = input_dim
        for level in range(int(levels)):
            blocks.append(
                TemporalBlock(
                    in_channels,
                    int(channels),
                    int(kernel_size),
                    dilation=2**level,
                    dropout=float(dropout),
                )
            )
            in_channels = int(channels)
        self.tcn = nn.Sequential(*blocks)
        self.asset_embedding = nn.Embedding(asset_count, asset_embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(int(channels) + asset_embedding_dim, int(channels)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(int(channels), 1),
        )

    def forward(self, x: torch.Tensor, asset_index: torch.Tensor) -> torch.Tensor:
        # DataLoader 输出是 [batch, time, feature]，Conv1d 需要 [batch, feature, time]。
        z = self.tcn(x.transpose(1, 2))[:, :, -1]
        embedding = self.asset_embedding(asset_index)
        return self.head(torch.cat([z, embedding], dim=1)).squeeze(-1)


def build_model(args: argparse.Namespace, input_dim: int, asset_count: int) -> nn.Module:
    if args.model == "mlp":
        return MLPResidualNet(
            input_dim=input_dim,
            asset_count=asset_count,
            asset_embedding_dim=args.asset_embedding_dim,
            hidden_dim=args.hidden_dim,
            layers=args.num_layers,
            dropout=args.dropout,
        )
    return TCNResidualNet(
        input_dim=input_dim,
        asset_count=asset_count,
        asset_embedding_dim=args.asset_embedding_dim,
        channels=args.tcn_channels,
        levels=args.tcn_levels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    )


def weighted_mse_loss(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.mean(weight * (prediction - target) ** 2)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        asset_index = batch["asset_index"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        weight = batch["weight"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            prediction = model(x, asset_index)
            loss = weighted_mse_loss(prediction, target, weight)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else np.nan


@torch.no_grad()
def predict_residual(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    row_indices = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        asset_index = batch["asset_index"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred = model(x, asset_index)
        predictions.append(pred.detach().cpu().numpy())
        row_indices.append(batch["row_index"].numpy())
    return np.concatenate(predictions), np.concatenate(row_indices)


def search_residual_weight_and_shrink(
    y_true: np.ndarray,
    weight: np.ndarray,
    asset_id: np.ndarray,
    time_id: np.ndarray,
    ridge_prediction: np.ndarray,
    residual_prediction: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    best = {"score": -np.inf}
    for residual_weight in np.arange(
        args.residual_weight_min,
        args.residual_weight_max + 1e-12,
        args.residual_weight_step,
    ):
        raw_prediction = ridge_prediction + float(residual_weight) * residual_prediction
        for cap in args.shrink_cap_candidates:
            shrink_info = calibrate_shrink_info(
                y_true,
                raw_prediction,
                weight,
                asset_id,
                args.shrink_mode,
                float(cap),
            )
            prediction = apply_shrink(raw_prediction, asset_id, shrink_info)
            score_info = score_candidate_on_calibration(
                y_true,
                prediction,
                weight,
                time_id,
                args.candidate_score_mode,
            )
            if score_info["selection_score"] > best["score"]:
                best = {
                    "score": float(score_info["selection_score"]),
                    "score_info": score_info,
                    "residual_weight": float(residual_weight),
                    "shrink_info": shrink_info,
                    "shrink_summary": summarize_shrink_info(shrink_info),
                    "prediction": prediction,
                    "raw_prediction": raw_prediction,
                }
    return best


def sample_tabular_dataset(
    x: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    asset_index: np.ndarray,
    row_index: np.ndarray,
    max_samples: int,
    seed: int,
) -> TabularResidualDataset:
    if max_samples and max_samples > 0 and len(row_index) > max_samples:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(np.arange(len(row_index)), size=int(max_samples), replace=False)
        chosen.sort()
        x = x[chosen]
        residual = residual[chosen]
        weight = weight[chosen]
        asset_index = asset_index[chosen]
        row_index = row_index[chosen]
    return TabularResidualDataset(x, residual, weight, asset_index, row_index)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.results_dir = make_results_dir(args.results_dir, args.model)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    frame, window_info, selected_features, fit_mask, cal_mask = load_window_frame(args)
    assets = sorted(frame["asset_id"].unique().astype(int).tolist())
    asset_to_index = {asset: index for index, asset in enumerate(assets)}

    fit_x_raw = frame.loc[fit_mask, selected_features].to_numpy(dtype=np.float32)
    cal_x_raw = frame.loc[cal_mask, selected_features].to_numpy(dtype=np.float32)
    fit_x, cal_x, _, _ = standardize(fit_x_raw, cal_x_raw)
    y_fit = frame.loc[fit_mask, "target"].to_numpy(dtype=np.float32)
    y_cal = frame.loc[cal_mask, "target"].to_numpy(dtype=np.float32)
    w_fit_raw = frame.loc[fit_mask, "weight"].to_numpy(dtype=np.float32)
    w_cal_raw = frame.loc[cal_mask, "weight"].to_numpy(dtype=np.float32)
    ridge_fit, ridge_cal, _ = fit_ridge_baseline(fit_x, cal_x, y_fit, w_fit_raw, args.ridge_alpha)
    residual_fit = y_fit - ridge_fit
    residual_cal = y_cal - ridge_cal

    # 全量标准化后的特征和 Ridge 预测，用全局行号对齐，便于 MLP/TCN 共享后续评估逻辑。
    x_all = np.zeros((len(frame), len(selected_features)), dtype=np.float32)
    residual_all = np.zeros(len(frame), dtype=np.float32)
    ridge_all = np.zeros(len(frame), dtype=np.float32)
    y_all = frame["target"].to_numpy(dtype=np.float32)
    weight_all_raw = frame["weight"].to_numpy(dtype=np.float32)
    weight_mean = max(float(np.mean(w_fit_raw)), 1e-12)
    weight_all = weight_all_raw / weight_mean
    asset_id_all = frame["asset_id"].to_numpy(dtype=np.int64)
    asset_index_all = np.asarray([asset_to_index[int(asset)] for asset in asset_id_all], dtype=np.int64)

    fit_positions = np.flatnonzero(fit_mask)
    cal_positions = np.flatnonzero(cal_mask)
    x_all[fit_positions] = fit_x
    x_all[cal_positions] = cal_x
    residual_all[fit_positions] = residual_fit
    residual_all[cal_positions] = residual_cal
    ridge_all[fit_positions] = ridge_fit
    ridge_all[cal_positions] = ridge_cal

    if args.model == "mlp":
        train_dataset = sample_tabular_dataset(
            fit_x,
            residual_fit,
            w_fit_raw / weight_mean,
            asset_index_all[fit_positions],
            fit_positions,
            args.max_train_samples,
            args.seed,
        )
        valid_dataset = TabularResidualDataset(
            cal_x,
            residual_cal,
            w_cal_raw / weight_mean,
            asset_index_all[cal_positions],
            cal_positions,
        )
    else:
        train_dataset = SequenceResidualDataset(
            frame,
            x_all,
            residual_all,
            weight_all,
            asset_to_index,
            args.sequence_len,
            fit_mask,
            args.max_train_samples,
            args.seed,
        )
        valid_dataset = SequenceResidualDataset(
            frame,
            x_all,
            residual_all,
            weight_all,
            asset_to_index,
            args.sequence_len,
            cal_mask,
            0,
            args.seed,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args, len(selected_features), len(assets)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp) if use_amp else None

    history = []
    best_state = None
    best_row = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        valid_residual_pred, valid_row_index = predict_residual(model, valid_loader, device, use_amp)
        y_valid = y_all[valid_row_index]
        w_valid = weight_all_raw[valid_row_index]
        asset_valid = asset_id_all[valid_row_index]
        time_valid = frame["time_id"].to_numpy(dtype=np.int64)[valid_row_index]
        ridge_valid = ridge_all[valid_row_index]
        best_epoch = search_residual_weight_and_shrink(
            y_valid,
            w_valid,
            asset_valid,
            time_valid,
            ridge_valid,
            valid_residual_pred,
            args,
        )
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "selection_score": float(best_epoch["score"]),
            "full_score": float(best_epoch["score_info"]["full_score"]),
            "first_half_score": float(best_epoch["score_info"]["first_half_score"]),
            "second_half_score": float(best_epoch["score_info"]["second_half_score"]),
            "residual_weight": float(best_epoch["residual_weight"]),
            "shrink": float(best_epoch["shrink_summary"]["cal_shrink"]),
            "shrink_mean": float(best_epoch["shrink_summary"]["cal_shrink_mean"]),
            "shrink_max": float(best_epoch["shrink_summary"]["cal_shrink_max"]),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, default=json_default))
        if best_row is None or row["selection_score"] > best_row["selection_score"]:
            best_row = row
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.early_stop_patience:
            print(f"early stop at epoch={epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    valid_residual_pred, valid_row_index = predict_residual(model, valid_loader, device, use_amp)
    y_valid = y_all[valid_row_index]
    w_valid = weight_all_raw[valid_row_index]
    asset_valid = asset_id_all[valid_row_index]
    time_valid = frame["time_id"].to_numpy(dtype=np.int64)[valid_row_index]
    ridge_valid = ridge_all[valid_row_index]
    final_best = search_residual_weight_and_shrink(
        y_valid,
        w_valid,
        asset_valid,
        time_valid,
        ridge_valid,
        valid_residual_pred,
        args,
    )
    final_prediction = final_best["prediction"]
    metrics = {
        "leakage_safe": True,
        "official_test_used": False,
        "device": str(device),
        "model": args.model,
        "config": vars(args),
        "window": window_info,
        "selected_feature_count": int(len(selected_features)),
        "train_samples": int(len(train_dataset)),
        "valid_samples": int(len(valid_dataset)),
        "ridge_only_raw_score": float(weighted_zero_mean_r2(y_valid, ridge_valid, w_valid)),
        "residual_raw_score": float(weighted_zero_mean_r2(y_valid - ridge_valid, valid_residual_pred, w_valid)),
        "final_selection_score": float(final_best["score"]),
        "final_full_score": float(final_best["score_info"]["full_score"]),
        "final_first_half_score": float(final_best["score_info"]["first_half_score"]),
        "final_second_half_score": float(final_best["score_info"]["second_half_score"]),
        "final_residual_weight": float(final_best["residual_weight"]),
        "final_shrink_summary": final_best["shrink_summary"],
        "final_shrink_info": final_best["shrink_info"],
        "prediction_std": float(np.std(final_prediction)),
        "residual_prediction_std": float(np.std(valid_residual_pred)),
    }
    metrics.update(score_time_blocks(y_valid, final_prediction, w_valid, time_valid, 4))
    metrics.update(score_time_blocks(y_valid, final_prediction, w_valid, time_valid, 8))

    pd.DataFrame(history).to_csv(args.results_dir / "training_history.csv", index=False)
    pd.DataFrame({"feature_name": selected_features}).to_csv(args.results_dir / "selected_features.csv", index=False)
    (args.results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    torch.save(model.state_dict(), args.results_dir / f"{args.model}_residual.pt")

    if args.save_validation_predictions:
        prediction_frame = frame.iloc[valid_row_index][["row_id", "time_id", "asset_id", "target", "weight"]].copy()
        prediction_frame["ridge_prediction"] = ridge_valid
        prediction_frame["residual_prediction"] = valid_residual_pred
        prediction_frame["prediction"] = final_prediction
        prediction_frame.to_csv(args.results_dir / "validation_predictions.csv", index=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
