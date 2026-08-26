from __future__ import annotations

import json
import os
import platform
import sys


# WSL exposes the host utility here, but it is not always on PATH inside Conda.
os.environ["PATH"] = "/usr/lib/wsl/lib:" + os.environ.get("PATH", "")


def check_torch() -> dict[str, object]:
    import torch

    result: dict[str, object] = {
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        result["device_name"] = torch.cuda.get_device_name(0)
        result["capability"] = list(torch.cuda.get_device_capability(0))
        x = torch.randn((2048, 2048), device="cuda")
        y = x @ x.T
        torch.cuda.synchronize()
        result["matmul_finite"] = bool(torch.isfinite(y).all().item())
    return result


def check_xgboost() -> dict[str, object]:
    import numpy as np
    import xgboost as xgb

    x = np.arange(128, dtype=np.float32).reshape(-1, 1)
    y = (x[:, 0] % 7).astype(np.float32)
    model = xgb.XGBRegressor(
        n_estimators=4,
        max_depth=2,
        tree_method="hist",
        device="cuda",
        n_jobs=1,
        verbosity=0,
    )
    model.fit(x, y)
    pred = model.predict(x[:8])
    return {
        "version": xgb.__version__,
        "device": str(model.get_xgb_params().get("device")),
        "prediction_finite": bool(np.isfinite(pred).all()),
    }


def check_catboost() -> dict[str, object]:
    import numpy as np
    from catboost import CatBoostRegressor

    x = np.arange(256, dtype=np.float32).reshape(-1, 2)
    y = (x[:, 0] * 0.1 - x[:, 1] * 0.03).astype(np.float32)
    model = CatBoostRegressor(
        iterations=4,
        depth=3,
        learning_rate=0.1,
        loss_function="RMSE",
        task_type="GPU",
        devices="0",
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x, y)
    pred = model.predict(x[:8])
    return {
        "version": str(model.get_metadata().get("catboost_version_info", "unknown")),
        "task_type": "GPU",
        "prediction_finite": bool(np.isfinite(pred).all()),
    }


def main() -> None:
    report: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    checks = {"torch": check_torch, "xgboost": check_xgboost, "catboost": check_catboost}
    failures: dict[str, str] = {}
    for name, check in checks.items():
        try:
            report[name] = check()
        except Exception as exc:  # Keep all checks visible in one run.
            failures[name] = f"{type(exc).__name__}: {exc}"
    report["failures"] = failures
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
