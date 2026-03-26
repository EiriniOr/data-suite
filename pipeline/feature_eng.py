from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
)
from config import HIGH_CARDINALITY_THRESHOLD


def engineer_features(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    scaler_type: str = "standard",  # "standard" | "minmax" | "robust" | "none"
    encoding: dict[str, str]
    | None = None,  # {col: "onehot" | "target" | "label" | "drop"}
    drop_high_corr: bool = True,
    corr_threshold: float = 0.95,
    # Time series options
    is_ts: bool = False,
    dt_col: str | None = None,
    lag_windows: list[int] | None = None,  # e.g. [1, 7, 30]
    rolling_windows: list[int] | None = None,  # e.g. [7, 30]
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Transform df into model-ready (X, y).

    Returns: (X, y, feature_report)
    """
    df = df.copy()
    report: dict = {"encoded": {}, "scaled": [], "dropped": [], "added": []}

    y = df[target_col].copy()
    df = df.drop(columns=[target_col])

    # ── Datetime feature extraction ───────────────────────────────────────────
    dt_cols = df.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns.tolist()
    if dt_col and dt_col in df.columns:
        dt_cols = list(set(dt_cols + [dt_col]))

    for col in dt_cols:
        try:
            s = pd.to_datetime(df[col])
            df[f"{col}_year"] = s.dt.year
            df[f"{col}_month"] = s.dt.month
            df[f"{col}_day"] = s.dt.day
            df[f"{col}_dayofweek"] = s.dt.dayofweek
            df[f"{col}_hour"] = s.dt.hour
            report["added"].extend(
                [
                    f"{col}_year",
                    f"{col}_month",
                    f"{col}_day",
                    f"{col}_dayofweek",
                    f"{col}_hour",
                ]
            )
            df = df.drop(columns=[col])
            report["dropped"].append(col)
        except Exception:
            pass

    # ── Time series lag + rolling features ────────────────────────────────────
    if is_ts and lag_windows:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        for col in numeric_cols[:3]:  # limit to first 3 numeric cols to avoid explosion
            for lag in lag_windows:
                new_col = f"{col}_lag{lag}"
                df[new_col] = df[col].shift(lag)
                report["added"].append(new_col)
        df = df.dropna()  # lags introduce nulls at the start

    if is_ts and rolling_windows:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        for col in [c for c in numeric_cols if "_lag" not in c][:3]:
            for w in rolling_windows:
                df[f"{col}_roll_mean_{w}"] = df[col].rolling(w).mean()
                df[f"{col}_roll_std_{w}"] = df[col].rolling(w).std()
                report["added"].extend([f"{col}_roll_mean_{w}", f"{col}_roll_std_{w}"])
        df = df.dropna()

    # ── Categorical encoding ──────────────────────────────────────────────────
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if encoding is None:
        encoding = _auto_encoding(df, cat_cols, y, task_type)

    # Target encoding (compute before one-hot to avoid shape issues)
    target_enc_map: dict[str, dict] = {}
    for col, strategy in encoding.items():
        if strategy == "target" and col in df.columns:
            target_mean = pd.concat([df[col], y], axis=1).groupby(col)[y.name].mean()
            target_enc_map[col] = target_mean.to_dict()
            df[col] = df[col].map(target_mean)
            report["encoded"][col] = "target_encoding"

    # One-hot encoding
    onehot_cols = [c for c, s in encoding.items() if s == "onehot" and c in df.columns]
    if onehot_cols:
        df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)
        report["encoded"].update({c: "one_hot" for c in onehot_cols})

    # Label encoding
    for col, strategy in encoding.items():
        if strategy == "label" and col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            report["encoded"][col] = "label_encoding"

    # Drop columns
    drop_cols = [c for c, s in encoding.items() if s == "drop" and c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        report["dropped"].extend(drop_cols)

    # ── Drop zero-variance features ───────────────────────────────────────────
    numeric = df.select_dtypes(include="number")
    zero_var = numeric.columns[numeric.std() == 0].tolist()
    if zero_var:
        df = df.drop(columns=zero_var)
        report["dropped"].extend(zero_var)

    # ── Drop highly correlated features ───────────────────────────────────────
    if drop_high_corr:
        to_drop = _corr_drop_list(df.select_dtypes(include="number"), corr_threshold)
        if to_drop:
            df = df.drop(columns=to_drop)
            report["dropped"].extend(to_drop)

    # ── Numeric scaling ───────────────────────────────────────────────────────
    numeric_cols_final = df.select_dtypes(include="number").columns.tolist()
    if scaler_type != "none" and numeric_cols_final:
        scaler_map = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        scaler = scaler_map.get(scaler_type, StandardScaler())
        df[numeric_cols_final] = scaler.fit_transform(df[numeric_cols_final])
        report["scaled"] = numeric_cols_final
        report["scaler_type"] = scaler_type

    # Align y index with df after potential row drops (TS lags)
    y = y.loc[df.index]

    return df, y, report


def _auto_encoding(
    df: pd.DataFrame, cat_cols: list[str], y: pd.Series, task_type: str
) -> dict:
    encoding = {}
    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique <= 2:
            encoding[col] = "label"
        elif n_unique <= HIGH_CARDINALITY_THRESHOLD:
            encoding[col] = "onehot"
        else:
            # High cardinality: target encoding for regression/classification, drop otherwise
            encoding[col] = (
                "target" if task_type in ("classification", "regression") else "drop"
            )
    return encoding


def _corr_drop_list(numeric_df: pd.DataFrame, threshold: float) -> list[str]:
    """Return list of columns to drop to remove correlation above threshold."""
    corr = numeric_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return to_drop
