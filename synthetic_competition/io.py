"""表格与 JSON 的读写工具，统一处理 CSV / Parquet 回退逻辑。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    # 所有输出目录都通过这个函数创建，避免到处重复写 mkdir 逻辑。
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def has_pyarrow() -> bool:
    # 只有安装了 pyarrow 时才优先使用 Parquet。
    try:
        import pyarrow  # noqa: F401
    except Exception:
        return False
    return True


def table_path(base_path: str | Path, prefer_parquet: bool = True) -> Path:
    base = Path(base_path)
    if base.suffix in {".csv", ".parquet"}:
        return base
    if prefer_parquet and has_pyarrow():
        return base.with_suffix(".parquet")
    return base.with_suffix(".csv")


def write_table(frame: pd.DataFrame, base_path: str | Path, prefer_parquet: bool = True) -> Path:
    # 根据环境决定写 CSV 还是 Parquet。
    path = table_path(base_path, prefer_parquet=prefer_parquet)
    ensure_dir(path.parent)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path


def resolve_table_path(base_path: str | Path) -> Path:
    base = Path(base_path)
    candidates = [base]
    if base.suffix == "":
        candidates.extend([base.with_suffix(".parquet"), base.with_suffix(".csv")])
    elif base.suffix == ".csv":
        candidates.append(base.with_suffix(".parquet"))
    elif base.suffix == ".parquet":
        candidates.append(base.with_suffix(".csv"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find table at {base_path}")


def read_table(base_path: str | Path) -> pd.DataFrame:
    # 读取时自动识别 CSV / Parquet 后缀。
    path = resolve_table_path(base_path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_json(payload: Any, path: str | Path) -> Path:
    # 统一写 JSON，保持 utf-8 和可读缩进。
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def read_json(path: str | Path) -> Any:
    # 统一从磁盘读取 JSON 配置或模型包。
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
