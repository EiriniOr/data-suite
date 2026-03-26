from __future__ import annotations
import pandas as pd
from sklearn.impute import KNNImputer
from config import NULL_DROP_THRESHOLD, OUTLIER_IQR_FACTOR


def clean(
    df: pd.DataFrame,
    null_strategies: dict[str, str] | None = None,
    outlier_action: str = "clip",  # "clip" | "drop" | "keep"
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Apply cleaning operations and return cleaned df + change report.

    Args:
        df: Input DataFrame.
        null_strategies: {col_name: strategy} where strategy is one of:
            "drop_col", "median", "mode", "zero", "knn", "keep"
        outlier_action: What to do with IQR outliers.
        drop_duplicates: Whether to drop duplicate rows.

    Returns:
        (cleaned_df, report) where report documents every change made.
    """
    df = df.copy()
    report: dict = {
        "dropped_cols": [],
        "imputed": {},
        "outliers": {},
        "duplicates_removed": 0,
    }

    # ── Duplicate rows ────────────────────────────────────────────────────────
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        report["duplicates_removed"] = before - len(df)

    # ── Null handling ─────────────────────────────────────────────────────────
    if null_strategies is None:
        null_strategies = auto_null_strategies(df)

    knn_cols = [c for c, s in null_strategies.items() if s == "knn"]
    if knn_cols:
        df = _knn_impute(df, knn_cols)
        for c in knn_cols:
            report["imputed"][c] = "knn"

    for col, strategy in null_strategies.items():
        if col not in df.columns or strategy in ("keep", "knn"):
            continue
        n_null = df[col].isna().sum()
        if n_null == 0:
            continue

        if strategy == "drop_col":
            df = df.drop(columns=[col])
            report["dropped_cols"].append(col)
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
            report["imputed"][col] = f"median ({df[col].median():.3g})"
        elif strategy == "mode":
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])
                report["imputed"][col] = f"mode ({mode_val.iloc[0]!r})"
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
            report["imputed"][col] = "zero"
        elif strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
            report["imputed"][col] = f"mean ({df[col].mean():.3g})"

    # ── Outlier handling ──────────────────────────────────────────────────────
    if outlier_action != "keep":
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - OUTLIER_IQR_FACTOR * iqr, q3 + OUTLIER_IQR_FACTOR * iqr
            n_outliers = ((df[col] < lo) | (df[col] > hi)).sum()
            if n_outliers == 0:
                continue
            if outlier_action == "clip":
                df[col] = df[col].clip(lo, hi)
                report["outliers"][col] = (
                    f"clipped {n_outliers} values to [{lo:.3g}, {hi:.3g}]"
                )
            elif outlier_action == "drop":
                mask = (df[col] >= lo) & (df[col] <= hi)
                df = df[mask]
                report["outliers"][col] = f"dropped {n_outliers} rows"

    return df, report


def auto_null_strategies(df: pd.DataFrame) -> dict[str, str]:
    """Generate sensible default null strategies for all columns with nulls."""
    strategies = {}
    for col in df.columns:
        null_pct = df[col].isna().sum() / len(df)
        if null_pct == 0:
            continue
        if null_pct > NULL_DROP_THRESHOLD:
            strategies[col] = "drop_col"
        elif pd.api.types.is_numeric_dtype(df[col]):
            skew = df[col].skew()
            strategies[col] = "median" if abs(skew) > 1 else "mean"
        else:
            strategies[col] = "mode"
    return strategies


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarizing outliers per numeric column."""
    rows = []
    for col in df.select_dtypes(include="number").columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - OUTLIER_IQR_FACTOR * iqr
        hi = q3 + OUTLIER_IQR_FACTOR * iqr
        n = ((df[col] < lo) | (df[col] > hi)).sum()
        pct = n / len(df) * 100
        rows.append(
            {
                "column": col,
                "n_outliers": n,
                "pct": round(pct, 2),
                "lower": round(lo, 4),
                "upper": round(hi, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("n_outliers", ascending=False)


def _knn_impute(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Apply KNN imputation to specified columns (numeric only)."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    target_cols = [c for c in cols if c in numeric_cols]
    if not target_cols:
        return df
    imputer = KNNImputer(n_neighbors=5)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df
