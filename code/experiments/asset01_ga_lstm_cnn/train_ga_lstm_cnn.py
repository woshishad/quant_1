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
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, Dataset


ASSET_IDS = (0, 1)


@dataclass
class ExperimentConfig:
    # 这里集中保存实验超参数，方便命令行只覆盖少数常用项，同时保证每次实验默认口径一致。
    seed: int = 42
    # 严格按时间切分：0..39999 训练，40000..49999 验证，避免未来信息泄露。
    train_end_time: int = 39_999
    valid_start_time: int = 40_000
    valid_end_time: int = 49_999
    # GA 每个个体是一个特征布尔掩码；这里限制入选特征数量，防止全选或选得太少。
    min_features: int = 16
    max_features: int = 96
    ga_population: int = 40
    ga_generations: int = 20
    ga_crossover: float = 0.8
    ga_mutation: float = 0.03
    ga_elitism: int = 2
    ridge_alpha: float = 10.0
    sequence_len: int = 256
    epochs: int = 3
    batch_size: int = 1024
    learning_rate: float = 3e-4
    lstm_hidden: int = 64
    cnn_channels: int = 64
    # asset embedding 让深度模型显式知道当前样本属于哪个标的。
    asset_embedding_dim: int = 4


def parse_args() -> argparse.Namespace:
    # 默认路径遵循项目整理后的目录约定：data 放数据，models 放模型，results 放指标和图。
    parser = argparse.ArgumentParser(description="Run GA feature selection + LSTM/CNN asset01 experiment.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/asset01_time50000"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/asset01_ga_lstm_cnn"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/asset01_ga_lstm_cnn"))
    parser.add_argument("--epochs", type=int, default=ExperimentConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=ExperimentConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=ExperimentConfig.learning_rate)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    # 固定随机种子，让 GA 初始种群、PyTorch 初始化和 DataLoader shuffle 尽量可复现。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_zero_mean_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    # 比赛使用的是零均值加权 R2：和“全部预测为 0”的基准相比，模型降低了多少加权平方误差。
    # 分数最高为 1，低于 0 表示还不如全部预测 0。
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    denominator = np.sum(weight * y_true * y_true)
    if denominator <= 0:
        return 0.0
    numerator = np.sum(weight * (y_true - y_pred) ** 2)
    return float(1.0 - numerator / denominator)


def equal_asset_weighted_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weight: np.ndarray,
    asset_id: np.ndarray,
) -> tuple[float, dict[str, float]]:
    # 每个 asset 内部使用原始 weight 算 R2；两个 asset 之间取算术平均，保证标的等权。
    scores = {}
    for asset in ASSET_IDS:
        mask = asset_id == asset
        scores[str(asset)] = weighted_zero_mean_r2(y_true[mask], y_pred[mask], weight[mask])
    return float(np.mean(list(scores.values()))), scores


def balanced_sample_weight(weight: np.ndarray, asset_id: np.ndarray) -> np.ndarray:
    # 训练时把每个 asset 的平均权重归一到 1，避免某个标的因权重尺度更大而主导优化。
    output = weight.astype(np.float64).copy()
    for asset in ASSET_IDS:
        mask = asset_id == asset
        mean_weight = float(np.mean(output[mask]))
        if mean_weight > 0:
            output[mask] /= mean_weight
    return output


def load_and_validate_dataset(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "train.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing dataset file: {path}")
    frame = pd.read_parquet(path)
    frame = frame.sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)

    if len(frame) != 100_000:
        raise ValueError(f"expected 100000 rows, got {len(frame)}")
    for asset in ASSET_IDS:
        asset_frame = frame.loc[frame["asset_id"] == asset]
        expected_times = np.arange(50_000)
        actual_times = asset_frame["time_id"].to_numpy(dtype=np.int64)
        if len(asset_frame) != 50_000 or not np.array_equal(actual_times, expected_times):
            raise ValueError(f"asset {asset} is not a 50000-point continuous time series")
    per_time_assets = frame.groupby("time_id")["asset_id"].nunique()
    if not bool((per_time_assets == 2).all()):
        raise ValueError("each time_id must contain exactly asset 0 and asset 1")
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col.startswith("feature_")]


def standardize_features(frame: pd.DataFrame, columns: list[str], train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = frame[columns].to_numpy(dtype=np.float32, copy=True)
    mean = np.nanmean(values[train_mask], axis=0).astype(np.float32)
    scale = np.nanstd(values[train_mask], axis=0).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    values = (values - mean) / scale
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return values, mean, scale


class GeneticFeatureSelector:
    # 遗传算法负责“选哪些 feature_* 给后续模型使用”。
    # 一个个体就是一个 bool mask，True 表示该特征被选中。
    # 适应度不是训练深度模型，而是训练一个 Ridge 代理模型，速度更快，适合在 GA 中反复评估。
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        w_train: np.ndarray,
        asset_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        w_valid: np.ndarray,
        asset_valid: np.ndarray,
        config: ExperimentConfig,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.w_train = balanced_sample_weight(w_train, asset_train)
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.w_valid = w_valid
        self.asset_valid = asset_valid
        self.config = config
        self.feature_count = x_train.shape[1]
        self.rng = np.random.default_rng(config.seed)
        self.cache: dict[bytes, float] = {}

    def repair(self, mask: np.ndarray) -> np.ndarray:
        # GA 的交叉和变异可能让特征数量过少或过多；repair 负责把个体拉回合法区间。
        mask = mask.astype(bool, copy=True)
        count = int(mask.sum())
        if count < self.config.min_features:
            candidates = np.flatnonzero(~mask)
            add = self.rng.choice(candidates, size=self.config.min_features - count, replace=False)
            mask[add] = True
        elif count > self.config.max_features:
            candidates = np.flatnonzero(mask)
            remove = self.rng.choice(candidates, size=count - self.config.max_features, replace=False)
            mask[remove] = False
        return mask

    def random_mask(self) -> np.ndarray:
        # 初始种群随机选择 min_features..max_features 个特征。
        k = int(self.rng.integers(self.config.min_features, self.config.max_features + 1))
        mask = np.zeros(self.feature_count, dtype=bool)
        mask[self.rng.choice(self.feature_count, size=k, replace=False)] = True
        return mask

    def fitness(self, mask: np.ndarray) -> float:
        # 同一个特征 mask 可能在不同代里反复出现，用 packbits 后的 bytes 做缓存 key，避免重复训练 Ridge。
        key = np.packbits(mask.astype(np.uint8)).tobytes()
        if key in self.cache:
            return self.cache[key]
        selected = np.flatnonzero(mask)
        # Ridge 只用当前个体选出的特征训练；sample_weight 已做 asset 内均值归一。
        model = Ridge(alpha=self.config.ridge_alpha, solver="lsqr", max_iter=200)
        model.fit(self.x_train[:, selected], self.y_train, sample_weight=self.w_train)
        pred = model.predict(self.x_valid[:, selected])
        score, _ = equal_asset_weighted_r2(self.y_valid, pred, self.w_valid, self.asset_valid)
        self.cache[key] = score
        return score

    def tournament(self, population: list[np.ndarray], scores: np.ndarray, size: int = 3) -> np.ndarray:
        # 锦标赛选择：随机抽几个个体，只让其中分数最高者繁殖，兼顾探索和优胜劣汰。
        indices = self.rng.choice(len(population), size=size, replace=False)
        return population[int(indices[np.argmax(scores[indices])])].copy()

    def crossover(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # 均匀交叉：对每个特征独立决定继承父代 A 还是父代 B。
        if self.rng.random() >= self.config.ga_crossover:
            return left.copy(), right.copy()
        mask = self.rng.random(self.feature_count) < 0.5
        child_a = np.where(mask, left, right)
        child_b = np.where(mask, right, left)
        return self.repair(child_a), self.repair(child_b)

    def mutate(self, mask: np.ndarray) -> np.ndarray:
        # 逐位变异：每个特征有 ga_mutation 的概率在选中/未选中之间翻转。
        flips = self.rng.random(self.feature_count) < self.config.ga_mutation
        mask = mask.copy()
        mask[flips] = ~mask[flips]
        return self.repair(mask)

    def run(self) -> tuple[np.ndarray, pd.DataFrame]:
        # 主循环记录每一代的最佳分、平均分、最佳个体特征数，方便画 GA fitness 曲线。
        population = [self.random_mask() for _ in range(self.config.ga_population)]
        history = []
        best_mask = population[0].copy()
        best_score = -np.inf

        for generation in range(self.config.ga_generations):
            scores = np.asarray([self.fitness(mask) for mask in population], dtype=np.float64)
            order = np.argsort(scores)[::-1]
            if scores[order[0]] > best_score:
                best_score = float(scores[order[0]])
                best_mask = population[int(order[0])].copy()
            history.append(
                {
                    "generation": generation,
                    "best_fitness": float(scores[order[0]]),
                    "mean_fitness": float(scores.mean()),
                    "selected_count": int(population[int(order[0])].sum()),
                    "global_best_fitness": best_score,
                }
            )
            print(
                f"GA generation {generation + 1}/{self.config.ga_generations}: "
                f"best={scores[order[0]]:.6f}, mean={scores.mean():.6f}, "
                f"selected={int(population[int(order[0])].sum())}"
            )

            next_population = [population[int(idx)].copy() for idx in order[: self.config.ga_elitism]]
            while len(next_population) < self.config.ga_population:
                parent_a = self.tournament(population, scores)
                parent_b = self.tournament(population, scores)
                child_a, child_b = self.crossover(parent_a, parent_b)
                next_population.append(self.mutate(child_a))
                if len(next_population) < self.config.ga_population:
                    next_population.append(self.mutate(child_b))
            population = next_population
        return best_mask, pd.DataFrame(history)


class TimeSeriesDataset(Dataset):
    # 把二维表格转换成深度模型需要的滑动时间窗口样本。
    # 每条样本形状是 [sequence_len, selected_feature_count]，标签是窗口最后一个时间点的 target。
    def __init__(
        self,
        frame: pd.DataFrame,
        x_selected: np.ndarray,
        sequence_len: int,
        split: str,
        config: ExperimentConfig,
    ):
        self.sequence_len = sequence_len
        self.groups = {}
        self.samples = []
        for asset in ASSET_IDS:
            # 每个 asset 单独建序列，避免 asset 0 的历史窗口混入 asset 1。
            asset_positions = np.flatnonzero(frame["asset_id"].to_numpy(dtype=np.int64) == asset)
            asset_frame = frame.iloc[asset_positions].reset_index(drop=True)
            self.groups[asset] = {
                "x": x_selected[asset_positions].astype(np.float32),
                "target": asset_frame["target"].to_numpy(dtype=np.float32),
                "weight": asset_frame["weight"].to_numpy(dtype=np.float32),
                "row_id": asset_frame["row_id"].to_numpy(dtype=np.int64),
                "time_id": asset_frame["time_id"].to_numpy(dtype=np.int64),
                "asset_id": asset_frame["asset_id"].to_numpy(dtype=np.int64),
            }
            for pos, time_id in enumerate(self.groups[asset]["time_id"]):
                if pos < sequence_len - 1:
                    # 前 sequence_len-1 个时间点没有足够历史窗口，因此不能作为样本标签。
                    continue
                if split == "train" and time_id <= config.train_end_time:
                    self.samples.append((asset, pos))
                elif split == "valid" and config.valid_start_time <= time_id <= config.valid_end_time:
                    self.samples.append((asset, pos))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        asset, pos = self.samples[index]
        group = self.groups[asset]
        start = pos + 1 - self.sequence_len
        # x 包含 [pos-sequence_len+1, pos] 的历史窗口；target/weight 对应 pos 这个最后时刻。
        x = group["x"][start : pos + 1]
        # 深度模型训练损失使用标的内归一后的权重，保持两个标的贡献接近。
        asset_weight_mean = float(np.mean(group["weight"]))
        train_weight = group["weight"][pos] / asset_weight_mean if asset_weight_mean > 0 else group["weight"][pos]
        return {
            "x": torch.from_numpy(x),
            "target": torch.tensor(group["target"][pos], dtype=torch.float32),
            "weight": torch.tensor(train_weight, dtype=torch.float32),
            "raw_weight": torch.tensor(group["weight"][pos], dtype=torch.float32),
            "row_id": torch.tensor(group["row_id"][pos], dtype=torch.int64),
            "time_id": torch.tensor(group["time_id"][pos], dtype=torch.int64),
            "asset_id": torch.tensor(group["asset_id"][pos], dtype=torch.int64),
        }


class LSTMRegressor(nn.Module):
    # LSTM 适合学习序列的长期依赖；这里额外拼接 asset embedding，让模型知道当前是哪一个标的。
    def __init__(self, input_dim: int, hidden_dim: int, asset_embedding_dim: int):
        super().__init__()
        self.asset_embedding = nn.Embedding(len(ASSET_IDS), asset_embedding_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim + asset_embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim + asset_embedding_dim, 1)

    def forward(self, x: torch.Tensor, asset_id: torch.Tensor) -> torch.Tensor:
        # 将 asset 身份显式喂给模型，避免两个标的被迫共享同一个截距和动态模式。
        asset_embed = self.asset_embedding(asset_id.long())
        repeated_embed = asset_embed.unsqueeze(1).expand(-1, x.shape[1], -1)
        x = torch.cat([x, repeated_embed], dim=-1)
        output, _ = self.lstm(x)
        features = torch.cat([output[:, -1, :], asset_embed], dim=-1)
        return self.head(features).squeeze(-1)


class CNNRegressor(nn.Module):
    # 1D-CNN 用卷积捕捉短期局部形态。这里使用左侧 padding 的“因果卷积”风格，避免卷积看到窗口右侧未来。
    def __init__(self, input_dim: int, channels: int, asset_embedding_dim: int):
        super().__init__()
        self.asset_embedding = nn.Embedding(len(ASSET_IDS), asset_embedding_dim)
        model_input_dim = input_dim + asset_embedding_dim
        self.net = nn.Sequential(
            nn.ConstantPad1d((4, 0), 0.0),
            nn.Conv1d(model_input_dim, channels, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.ConstantPad1d((4, 0), 0.0),
            nn.Conv1d(channels, channels // 2, kernel_size=3, dilation=2, padding=0),
            nn.ReLU(),
        )
        self.head = nn.Linear(channels // 2 + asset_embedding_dim, 1)

    def forward(self, x: torch.Tensor, asset_id: torch.Tensor) -> torch.Tensor:
        # 因果卷积只看当前和历史窗口，最后取最新时刻特征，不再做全局平均池化。
        asset_embed = self.asset_embedding(asset_id.long())
        repeated_embed = asset_embed.unsqueeze(1).expand(-1, x.shape[1], -1)
        x = torch.cat([x, repeated_embed], dim=-1)
        x = x.transpose(1, 2)
        features = self.net(x)[:, :, -1]
        features = torch.cat([features, asset_embed], dim=-1)
        return self.head(features).squeeze(-1)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float], pd.DataFrame]:
    # 统一评估函数：跑完整个验证集，返回等权 asset R2、分 asset R2 和逐行预测明细。
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            asset_id = batch["asset_id"].to(device)
            pred = model(x, asset_id).cpu().numpy()
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
    score, by_asset = equal_asset_weighted_r2(
        frame["target"].to_numpy(),
        frame["prediction"].to_numpy(),
        frame["weight"].to_numpy(),
        frame["asset_id"].to_numpy(),
    )
    return score, by_asset, frame


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    name: str,
) -> tuple[nn.Module, dict, pd.DataFrame]:
    # 深度模型训练流程：
    # 1. 用加权 MSE 做优化目标；
    # 2. 每个 epoch 后在验证集上计算最终指标；
    # 3. 保存验证分数最好的参数，而不是最后一个 epoch 的参数。
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_score = -np.inf
    best_pred = pd.DataFrame()
    history = []

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

        valid_score, valid_by_asset, valid_pred = evaluate_model(model, valid_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_rows, 1),
                "valid_equal_asset_weighted_r2": valid_score,
                "valid_asset_scores": valid_by_asset,
            }
        )
        print(
            f"{name} epoch {epoch + 1}/{config.epochs}: "
            f"loss={history[-1]['train_loss']:.6f}, valid={valid_score:.6f}"
        )
        if valid_score > best_score:
            best_score = valid_score
            best_state = copy.deepcopy(model.state_dict())
            best_pred = valid_pred.copy()

    model.load_state_dict(best_state)
    final_score, final_by_asset, final_pred = evaluate_model(model, valid_loader, device)
    if best_pred.empty or final_score >= best_score:
        best_pred = final_pred
        best_score = final_score
    return model, {"best_score": best_score, "by_asset": final_by_asset, "history": history}, best_pred


def fit_ridge_baseline(
    x_selected: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    asset_id: np.ndarray,
    train_mask: np.ndarray,
    valid_sequences: TimeSeriesDataset,
    config: ExperimentConfig,
) -> tuple[dict, pd.DataFrame, Ridge]:
    # Ridge 作为第三个可融合模型，沿用 GA 的轻量线性视角，并对验证序列对应行出预测。
    # 注意：Ridge 不吃历史窗口，只使用“窗口最后一个时间点”的 selected features。
    # 这样它和 LSTM/CNN 的预测 row_id 完全对齐，可以直接做融合。
    sample_weight = balanced_sample_weight(weight[train_mask], asset_id[train_mask])
    model = Ridge(alpha=config.ridge_alpha, solver="lsqr", max_iter=500)
    model.fit(x_selected[train_mask], y[train_mask], sample_weight=sample_weight)

    rows = []
    for asset, pos in valid_sequences.samples:
        group = valid_sequences.groups[asset]
        prediction = float(model.predict(group["x"][pos : pos + 1])[0])
        rows.append(
            {
                "row_id": int(group["row_id"][pos]),
                "time_id": int(group["time_id"][pos]),
                "asset_id": int(group["asset_id"][pos]),
                "target": float(group["target"][pos]),
                "weight": float(group["weight"][pos]),
                "prediction": prediction,
            }
        )
    pred_frame = pd.DataFrame(rows).sort_values(["time_id", "asset_id"], kind="mergesort").reset_index(drop=True)
    score, by_asset = equal_asset_weighted_r2(
        pred_frame["target"].to_numpy(),
        pred_frame["prediction"].to_numpy(),
        pred_frame["weight"].to_numpy(),
        pred_frame["asset_id"].to_numpy(),
    )
    return {"score": float(score), "by_asset": by_asset}, pred_frame, model


def find_best_blend(
    ridge_pred: pd.DataFrame,
    lstm_pred: pd.DataFrame,
    cnn_pred: pd.DataFrame,
) -> tuple[dict[str, float], float, dict[str, float], pd.DataFrame]:
    # 三模型融合权重使用简单网格搜索：
    # ridge_weight + lstm_weight + cnn_weight = 1，且三个权重都非负。
    # 目标函数仍然是两个 asset 等权平均的 weighted zero-mean R2。
    merged = ridge_pred.rename(columns={"prediction": "ridge_prediction"}).merge(
        lstm_pred[["row_id", "prediction"]].rename(columns={"prediction": "lstm_prediction"}),
        on="row_id",
        how="inner",
    )
    merged = merged.merge(
        cnn_pred[["row_id", "prediction"]].rename(columns={"prediction": "cnn_prediction"}),
        on="row_id",
        how="inner",
    )
    best_weights = {"ridge": 0.0, "lstm": 0.0, "cnn": 0.0}
    best_score = -np.inf
    best_by_asset = {}
    prediction_columns = {
        "ridge": merged["ridge_prediction"].to_numpy(),
        "lstm": merged["lstm_prediction"].to_numpy(),
        "cnn": merged["cnn_prediction"].to_numpy(),
    }
    grid = np.linspace(0.0, 1.0, 21)
    for ridge_weight in grid:
        for lstm_weight in grid:
            cnn_weight = 1.0 - ridge_weight - lstm_weight
            if cnn_weight < -1e-12:
                continue
            cnn_weight = max(0.0, cnn_weight)
            prediction = (
                ridge_weight * prediction_columns["ridge"]
                + lstm_weight * prediction_columns["lstm"]
                + cnn_weight * prediction_columns["cnn"]
            )
            score, by_asset = equal_asset_weighted_r2(
                merged["target"].to_numpy(),
                prediction,
                merged["weight"].to_numpy(),
                merged["asset_id"].to_numpy(),
            )
            if score > best_score:
                best_weights = {"ridge": float(ridge_weight), "lstm": float(lstm_weight), "cnn": float(cnn_weight)}
                best_score = float(score)
                best_by_asset = by_asset
    merged["prediction"] = (
        best_weights["ridge"] * prediction_columns["ridge"]
        + best_weights["lstm"] * prediction_columns["lstm"]
        + best_weights["cnn"] * prediction_columns["cnn"]
    )
    merged["error"] = merged["prediction"] - merged["target"]
    return best_weights, best_score, best_by_asset, merged


def save_plots(results_dir: Path, ga_history: pd.DataFrame, predictions: pd.DataFrame, metrics: dict) -> None:
    # 结果图只用于实验诊断，不参与训练：
    # - GA 曲线看因子选择是否收敛；
    # - target vs prediction 看预测是否塌缩或偏移；
    # - score_by_asset 看两个标的是否被某一边拖累。
    plt.figure(figsize=(8, 5))
    plt.plot(ga_history["generation"], ga_history["best_fitness"], label="generation best")
    plt.plot(ga_history["generation"], ga_history["mean_fitness"], label="generation mean")
    plt.xlabel("generation")
    plt.ylabel("equal-asset weighted R2")
    plt.title("GA Fitness Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "ga_fitness_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(predictions["target"], predictions["prediction"], s=7, alpha=0.25)
    limit = float(np.nanmax(np.abs(predictions[["target", "prediction"]].to_numpy())))
    limit = max(limit, 1e-6)
    plt.plot([-limit, limit], [-limit, limit], color="black", linewidth=1)
    plt.xlabel("target")
    plt.ylabel("prediction")
    plt.title(f"Target vs Prediction, score={metrics['blend']['score']:.4f}")
    plt.tight_layout()
    plt.savefig(results_dir / "target_vs_prediction.png", dpi=160)
    plt.close()

    asset_labels = ["0", "1"]
    x = np.arange(len(asset_labels))
    width = 0.25
    plt.figure(figsize=(8, 5))
    for offset, key in [(-1.5 * width, "ridge"), (-0.5 * width, "lstm"), (0.5 * width, "cnn"), (1.5 * width, "blend")]:
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
    config = ExperimentConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
    set_seed(config.seed)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # 加载 build_dataset.py 生成的 10 万行连续面板，并确认基础口径无误。
    frame = load_and_validate_dataset(args.data_dir)
    columns = feature_columns(frame)
    # 训练/验证严格按 time_id 切分，不能随机切分，否则会把未来时间段的信息泄露给训练集。
    train_mask = frame["time_id"].to_numpy(dtype=np.int64) <= config.train_end_time
    valid_mask = (frame["time_id"].to_numpy(dtype=np.int64) >= config.valid_start_time) & (
        frame["time_id"].to_numpy(dtype=np.int64) <= config.valid_end_time
    )
    x_all, mean, scale = standardize_features(frame, columns, train_mask)
    y = frame["target"].to_numpy(dtype=np.float32)
    weight = frame["weight"].to_numpy(dtype=np.float32)
    asset_id = frame["asset_id"].to_numpy(dtype=np.int64)

    # 第一步：GA + Ridge 代理模型选因子。GA 只决定 selected_features，不直接训练最终深度模型。
    selector = GeneticFeatureSelector(
        x_train=x_all[train_mask],
        y_train=y[train_mask],
        w_train=weight[train_mask],
        asset_train=asset_id[train_mask],
        x_valid=x_all[valid_mask],
        y_valid=y[valid_mask],
        w_valid=weight[valid_mask],
        asset_valid=asset_id[valid_mask],
        config=config,
    )
    selected_mask, ga_history = selector.run()
    selected_indices = np.flatnonzero(selected_mask)
    selected_features = [columns[idx] for idx in selected_indices]
    if not (config.min_features <= len(selected_features) <= config.max_features):
        raise ValueError(f"GA selected {len(selected_features)} features, outside expected bounds")

    # 第二步：只保留 GA 选出的特征，并构造 LSTM/CNN 需要的滑动窗口 Dataset。
    x_selected = x_all[:, selected_indices]
    train_dataset = TimeSeriesDataset(frame, x_selected, config.sequence_len, "train", config)
    valid_dataset = TimeSeriesDataset(frame, x_selected, config.sequence_len, "valid", config)
    if len(valid_dataset) == 0:
        raise ValueError("no validation sequences were built")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device={device}, selected_features={len(selected_features)}")

    # 第三步：训练 Ridge 基线。它既是独立对照，也是最终融合的一路输入。
    ridge_metrics, ridge_pred, ridge_model = fit_ridge_baseline(
        x_selected=x_selected,
        y=y,
        weight=weight,
        asset_id=asset_id,
        train_mask=train_mask,
        valid_sequences=valid_dataset,
        config=config,
    )
    print(f"Ridge valid={ridge_metrics['score']:.6f}")

    # 第四步：训练两个非线性时序模型。
    # LSTM 更偏长短期记忆，CNN 更偏局部形态和近期变化；二者都带 asset embedding。
    lstm = LSTMRegressor(
        input_dim=len(selected_features),
        hidden_dim=config.lstm_hidden,
        asset_embedding_dim=config.asset_embedding_dim,
    )
    cnn = CNNRegressor(
        input_dim=len(selected_features),
        channels=config.cnn_channels,
        asset_embedding_dim=config.asset_embedding_dim,
    )
    lstm, lstm_metrics, lstm_pred = train_model(lstm, train_loader, valid_loader, device, config, "LSTM")
    cnn, cnn_metrics, cnn_pred = train_model(cnn, train_loader, valid_loader, device, config, "CNN")

    # 第五步：把 Ridge/LSTM/CNN 三路预测按验证集指标搜索最优非负融合权重。
    blend_weights, blend_score, blend_by_asset, predictions = find_best_blend(ridge_pred, lstm_pred, cnn_pred)
    metrics = {
        "config": asdict(config),
        "device": str(device),
        "selected_feature_count": len(selected_features),
        "train_sequences": len(train_dataset),
        "valid_sequences": len(valid_dataset),
        "ga_best_fitness": float(ga_history["global_best_fitness"].max()),
        "ridge": {"score": float(ridge_metrics["score"]), "by_asset": ridge_metrics["by_asset"]},
        "lstm": {"score": float(lstm_metrics["best_score"]), "by_asset": lstm_metrics["by_asset"]},
        "cnn": {"score": float(cnn_metrics["best_score"]), "by_asset": cnn_metrics["by_asset"]},
        "blend": {"score": float(blend_score), "by_asset": blend_by_asset, "weights": blend_weights},
    }

    # 保存模型和元数据。metadata 里记录选中特征、标准化参数、Ridge 参数和本次实验指标。
    torch.save({"state_dict": lstm.state_dict(), "input_dim": len(selected_features), "config": asdict(config)}, args.model_dir / "lstm.pt")
    torch.save({"state_dict": cnn.state_dict(), "input_dim": len(selected_features), "config": asdict(config)}, args.model_dir / "cnn.pt")
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

    # 保存所有可检查结果：逐行预测、选中特征、GA 历史、最终指标和诊断图。
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
