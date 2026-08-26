"""命令行入口：加载训练好的模型包，对测试集执行推理并输出提交文件。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_competition.models import Standardizer
from synthetic_competition.io import read_json, read_table
from synthetic_competition.models import FeatureInteractionBuilder, RidgeRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for the synthetic competition model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/model_bundle.json"))
    parser.add_argument("--output-path", type=Path, default=Path("submission.csv"))
    return parser.parse_args()


def load_target_model(bundle: dict) -> tuple[list[str], RidgeRegressor, RidgeRegressor, FeatureInteractionBuilder, float]:
    # 把 JSON 里的参数重新装回可预测的模型对象。
    feature_names = bundle["feature_names"]
    target_model = bundle["target_model"]

    base = RidgeRegressor(alpha=target_model["base_model"]["alpha"])
    base.standardizer = Standardizer(
        mean_=np.asarray(target_model["base_model"]["mean"], dtype=float),
        scale_=np.asarray(target_model["base_model"]["scale"], dtype=float),
    )
    base.coef_ = np.asarray(target_model["base_model"]["coef"], dtype=float)
    base.intercept_ = float(target_model["base_model"]["intercept"])

    interaction = RidgeRegressor(alpha=target_model["interaction_model"]["alpha"])
    interaction.standardizer = Standardizer(
        mean_=np.asarray(target_model["interaction_model"]["mean"], dtype=float),
        scale_=np.asarray(target_model["interaction_model"]["scale"], dtype=float),
    )
    interaction.coef_ = np.asarray(target_model["interaction_model"]["coef"], dtype=float)
    interaction.intercept_ = float(target_model["interaction_model"]["intercept"])

    builder = FeatureInteractionBuilder(tuple(target_model["interaction_features"]), include_squares=True)
    base_weight = float(target_model["base_weight"])
    return feature_names, base, interaction, builder, base_weight


def predict_from_bundle(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    # 先算线性主项，再算交互项，最后按权重融合。
    feature_names, base, interaction, builder, base_weight = load_target_model(bundle)
    X = frame[feature_names].to_numpy(dtype=float)
    base_pred = base.predict(X)
    interaction_features, _ = builder.transform(X)
    interaction_pred = interaction.predict(interaction_features)
    return base_weight * base_pred + (1.0 - base_weight) * interaction_pred


def main() -> None:
    # 读取测试集、完成推理并写出提交文件。
    args = parse_args()
    bundle = read_json(args.model_path)
    test_frame = read_table(args.data_dir / "test")
    predictions = predict_from_bundle(bundle, test_frame)
    submission = pd.DataFrame({"row_id": test_frame["row_id"].to_numpy(), "target": predictions.astype(float)})
    submission.to_csv(args.output_path, index=False)
    print(args.output_path)


if __name__ == "__main__":
    main()
