from __future__ import annotations
import pandas as pd
from dataclasses import dataclass, field
from utils.ts_detector import detect_time_series


@dataclass
class DatasetMeta:
    """Metadata that flows between pipeline stages."""

    filename: str = ""
    n_rows: int = 0
    n_cols: int = 0
    task_type: str | None = None  # "classification" | "regression" | "time_series"
    target_col: str | None = None
    datetime_col: str | None = None
    is_ts: bool = False
    ts_freq: str | None = None
    approved_workflow: list[str] = field(default_factory=list)
    cleaning_report: dict = field(default_factory=dict)
    feature_report: dict = field(default_factory=dict)
    best_model_name: str | None = None
    best_model_score: float | None = None
    best_model_metric: str | None = None

    def to_dict(self) -> dict:
        return {
            k: v
            for k, v in self.__dict__.items()
            if v is not None and v != [] and v != {}
        }


def load_file(uploaded_file) -> tuple[pd.DataFrame, DatasetMeta]:
    """Load an uploaded file into a DataFrame + initial DatasetMeta.

    Accepts: CSV, Excel (.xlsx/.xls), Parquet.
    Returns cleaned dtypes (numeric coercion, datetime parsing attempted).
    """
    name = uploaded_file.name
    ext = name.rsplit(".", 1)[-1].lower()

    if ext == "csv":
        df = pd.read_csv(uploaded_file)
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(uploaded_file)
    elif ext == "parquet":
        df = pd.read_parquet(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Use CSV, Excel, or Parquet.")

    df = _coerce_types(df)
    ts_info = detect_time_series(df)

    meta = DatasetMeta(
        filename=name,
        n_rows=len(df),
        n_cols=len(df.columns),
        is_ts=ts_info["is_ts"],
        datetime_col=ts_info["datetime_col"],
        ts_freq=ts_info["freq"],
    )
    return df, meta


def infer_task_type(df: pd.DataFrame, target_col: str) -> str:
    """Infer classification vs regression from the target column.

    Rules:
    - ≤ 20 unique values or object/bool dtype → classification
    - Numeric with > 20 unique values → regression
    """
    series = df[target_col]
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series):
        return "classification"
    n_unique = series.nunique()
    if n_unique <= 20:
        return "classification"
    return "regression"


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to improve dtype inference from a freshly loaded DataFrame."""
    for col in df.columns:
        # Numeric coercion for object cols that look numeric
        if df[col].dtype == object:
            try:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() / len(df) > 0.8:
                    df[col] = converted
                    continue
            except Exception:
                pass
            # Datetime coercion
            if _looks_like_datetime(df[col]):
                try:
                    df[col] = pd.to_datetime(
                        df[col], infer_datetime_format=True, errors="coerce"
                    )
                except Exception:
                    pass
    return df


def _looks_like_datetime(series: pd.Series) -> bool:
    """Heuristic: check if a handful of values parse as datetime."""
    sample = series.dropna().head(10).astype(str)
    hits = 0
    for val in sample:
        try:
            pd.to_datetime(val)
            hits += 1
        except Exception:
            pass
    return hits / max(len(sample), 1) > 0.6
