from __future__ import annotations
import pandas as pd
import numpy as np
from utils.plot_utils import (
    distribution_plot,
    correlation_heatmap,
    class_balance_chart,
    time_series_plot,
)


def compute_summary(df: pd.DataFrame) -> dict:
    """Compute descriptive statistics summary."""
    numeric = df.select_dtypes(include="number")
    desc = numeric.describe().T
    desc["skew"] = numeric.skew()
    desc["kurtosis"] = numeric.kurtosis()

    categorical = df.select_dtypes(include=["object", "category", "bool"])
    cat_summary = {}
    for col in categorical.columns:
        vc = df[col].value_counts()
        cat_summary[col] = {
            "unique": df[col].nunique(),
            "top": vc.index[0] if len(vc) > 0 else None,
            "top_freq": int(vc.iloc[0]) if len(vc) > 0 else 0,
            "null_pct": round(df[col].isna().mean() * 100, 2),
        }

    return {
        "shape": df.shape,
        "numeric_stats": desc,
        "categorical_stats": cat_summary,
        "null_counts": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def get_distribution_figures(df: pd.DataFrame, cols: list[str] | None = None):
    """Return {col: plotly_fig} for numeric columns."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if cols:
        numeric_cols = [c for c in cols if c in numeric_cols]
    return {col: distribution_plot(df[col]) for col in numeric_cols}


def get_correlation_figure(df: pd.DataFrame):
    return correlation_heatmap(df)


def get_target_distribution(df: pd.DataFrame, target_col: str, task_type: str):
    if task_type == "classification":
        return class_balance_chart(df[target_col])
    else:
        return distribution_plot(df[target_col])


def get_ts_figures(df: pd.DataFrame, dt_col: str, value_cols: list[str]):
    """Return time series line charts for each value column."""
    return {col: time_series_plot(df, dt_col, col) for col in value_cols}


def flag_issues(
    df: pd.DataFrame, target_col: str | None = None, task_type: str | None = None
) -> list[str]:
    """Return a list of human-readable data quality warnings."""
    warnings = []

    # High null columns
    null_pct = df.isnull().mean() * 100
    high_null = null_pct[null_pct > 20]
    for col, pct in high_null.items():
        warnings.append(f"**{col}** has {pct:.1f}% missing values.")

    # High correlation pairs
    numeric = df.select_dtypes(include="number")
    if len(numeric.columns) > 1:
        corr = numeric.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr = [
            (c1, c2, upper.loc[c1, c2])
            for c1 in upper.columns
            for c2 in upper.index
            if upper.loc[c2, c1] > 0.95
        ]
        for c1, c2, val in high_corr[:5]:
            warnings.append(
                f"**{c1}** and **{c2}** are {val:.2f} correlated — consider dropping one."
            )

    # Class imbalance
    if task_type == "classification" and target_col and target_col in df.columns:
        vc = df[target_col].value_counts(normalize=True)
        minority_pct = vc.min() * 100
        if minority_pct < 10:
            warnings.append(
                f"Severe class imbalance: minority class is {minority_pct:.1f}% of data. "
                "Consider SMOTE or `class_weight='balanced'`."
            )
        elif minority_pct < 30:
            warnings.append(
                f"Mild class imbalance: minority class is {minority_pct:.1f}%."
            )

    # Low variance columns
    for col in numeric.columns:
        if numeric[col].std() == 0:
            warnings.append(
                f"**{col}** has zero variance — will be dropped in feature engineering."
            )

    return warnings
