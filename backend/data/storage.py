"""Local CSV storage for market data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.data.config import get_settings
from backend.data.schemas import SCHEMAS


class DataStorage:
    """Simple storage layer using CSV files, partitioned by table and optional partition key."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = (Path(root) if root is not None else get_settings().db_path.parent).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _table_path(self, table: str, partition: str | None = None) -> Path:
        if partition:
            return self._root / table / f"{partition}.csv"
        return self._root / f"{table}.csv"

    def save(self, df: pd.DataFrame, table: str, partition: str | None = None) -> None:
        path = self._table_path(table, partition)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")

    def load(self, table: str, partition: str | None = None) -> pd.DataFrame:
        path = self._table_path(table, partition)
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        schema = SCHEMAS.get(table)
        if schema is None or df.empty:
            return df

        result = pd.DataFrame(index=df.index)
        for spec in schema.columns:
            if spec.name not in df.columns:
                if spec.required:
                    result[spec.name] = pd.NA
                continue
            if spec.dtype in ("float64", "int64"):
                result[spec.name] = pd.to_numeric(df[spec.name], errors="coerce")
            else:
                result[spec.name] = df[spec.name].astype("string")
        return result

    def exists(self, table: str, partition: str | None = None) -> bool:
        return self._table_path(table, partition).exists()

    def delete(self, table: str, partition: str | None = None) -> None:
        path = self._table_path(table, partition)
        if path.exists():
            path.unlink()
