"""Causal inference — ATE via propensity score matching and IPW.

No external causal libraries required — built on scikit-learn + scipy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go


_DARK = {
    "template": "plotly_dark",
    "paper_bgcolor": "#07071a",
    "plot_bgcolor": "#0f0f2d",
}


def run_causal(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    covariate_cols: list[str],
    method: str = "psm",  # "psm" | "ipw"
) -> dict:
    """
    Estimate the Average Treatment Effect (ATE) on observational data.

    method="psm"  — propensity score matching (1:1 nearest neighbor)
    method="ipw"  — inverse probability weighting

    Returns metrics, balance table, and plotly figures.
    """
    if not covariate_cols:
        raise ValueError("Select at least one covariate (confounder) to adjust for.")

    df = df.copy().reset_index(drop=True)

    groups = df[treatment_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError(
            f"Treatment must be binary. Found: {sorted(str(g) for g in groups)}"
        )

    groups_sorted = sorted(groups, key=str)
    ctrl_label, trt_label = groups_sorted[0], groups_sorted[1]
    df["_T"] = (df[treatment_col] == trt_label).astype(int)

    # Encode covariates
    X_cov = df[covariate_cols].copy()
    X_cov = pd.get_dummies(X_cov, drop_first=True).astype(float)
    X_cov = X_cov.fillna(X_cov.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cov)

    T = df["_T"].values
    Y = df[outcome_col].values.astype(float)

    # Naive (unadjusted) difference
    naive_diff = float(Y[T == 1].mean() - Y[T == 0].mean())

    # ── Propensity scores ─────────────────────────────────────────────────────
    ps_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    ps_model.fit(X_scaled, T)
    ps = ps_model.predict_proba(X_scaled)[:, 1]
    df["_ps"] = ps

    results: dict = {
        "control_label": str(ctrl_label),
        "treatment_label": str(trt_label),
        "n_treatment": int(T.sum()),
        "n_control": int((1 - T).sum()),
        "naive_diff": round(naive_diff, 4),
        "figures": {},
    }

    if method == "psm":
        ate, ate_se, matched_df = _psm(df, T, Y, ps)
        results["method"] = "Propensity Score Matching (1:1 nearest neighbor)"
    else:
        ate, ate_se, matched_df = _ipw(T, Y, ps)
        results["method"] = "Inverse Probability Weighting (IPW)"
        matched_df = None

    ci = (ate - 1.96 * ate_se, ate + 1.96 * ate_se)
    significant = not (ci[0] < 0 < ci[1])  # CI doesn't cross zero
    bias_reduction = (
        abs(ate - naive_diff) / abs(naive_diff) * 100 if naive_diff != 0 else 0.0
    )

    balance = _covariate_balance(df, covariate_cols, matched_df)

    results.update(
        {
            "ate": round(float(ate), 4),
            "ate_ci": (round(float(ci[0]), 4), round(float(ci[1]), 4)),
            "ate_se": round(float(ate_se), 4),
            "significant": significant,
            "bias_reduction_pct": round(float(bias_reduction), 1),
            "balance_stats": balance,
            "matched_n": len(matched_df) if matched_df is not None else None,
            "outcome_col": outcome_col,
            "treatment_col": treatment_col,
        }
    )

    # Figures
    results["figures"]["propensity"] = _propensity_fig(df, ctrl_label, trt_label)
    results["figures"]["balance"] = _balance_fig(balance)
    results["figures"]["effect"] = _effect_fig(
        naive_diff, ate, ci, ctrl_label, trt_label, outcome_col
    )

    return results


# ── Estimators ────────────────────────────────────────────────────────────────


def _psm(df, T, Y, ps) -> tuple[float, float, pd.DataFrame]:
    """1:1 nearest-neighbor matching without replacement."""
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    matched_t, matched_c = [], []
    used = set()
    for ti in treated_idx:
        dist = np.abs(ps[control_idx] - ps[ti])
        for rank in np.argsort(dist):
            ci = control_idx[rank]
            if ci not in used:
                matched_t.append(ti)
                matched_c.append(ci)
                used.add(ci)
                break

    if not matched_t:
        raise ValueError(
            "No matches found. Check that treatment and control groups have "
            "overlapping propensity scores (common support)."
        )

    Y_t = Y[matched_t]
    Y_c = Y[matched_c]
    ate = float(Y_t.mean() - Y_c.mean())

    # Bootstrap SE (200 resamples)
    rng = np.random.default_rng(42)
    n = len(matched_t)
    boot_ates = []
    for _ in range(200):
        idx = rng.integers(0, n, size=n)
        boot_ates.append(Y_t[idx].mean() - Y_c[idx].mean())
    ate_se = float(np.std(boot_ates))

    matched_df = df.iloc[matched_t + matched_c].copy()
    return ate, ate_se, matched_df


def _ipw(T, Y, ps) -> tuple[float, float, None]:
    """Horvitz-Thompson IPW estimator with trimming."""
    eps = 0.01
    mask = (ps >= eps) & (ps <= 1 - eps)
    T, Y, ps = T[mask], Y[mask], ps[mask]

    weights = T / ps - (1 - T) / (1 - ps)
    ate = float(np.mean(weights * Y))

    rng = np.random.default_rng(42)
    n = len(T)
    boot_ates = []
    for _ in range(200):
        idx = rng.integers(0, n, size=n)
        w = T[idx] / ps[idx] - (1 - T[idx]) / (1 - ps[idx])
        boot_ates.append(float(np.mean(w * Y[idx])))
    ate_se = float(np.std(boot_ates))

    return ate, ate_se, None


# ── Balance ───────────────────────────────────────────────────────────────────


def _covariate_balance(df, covariate_cols, matched_df) -> pd.DataFrame:
    rows = []
    for col in covariate_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        t = df[df["_T"] == 1][col]
        c = df[df["_T"] == 0][col]
        pooled_std = np.sqrt((t.var() + c.var()) / 2)
        smd_before = (
            float((t.mean() - c.mean()) / pooled_std) if pooled_std > 0 else 0.0
        )

        smd_after = smd_before
        if (
            matched_df is not None
            and col in matched_df.columns
            and "_T" in matched_df.columns
        ):
            tm = matched_df[matched_df["_T"] == 1][col]
            cm = matched_df[matched_df["_T"] == 0][col]
            ps_m = np.sqrt((tm.var() + cm.var()) / 2)
            smd_after = float((tm.mean() - cm.mean()) / ps_m) if ps_m > 0 else 0.0

        rows.append(
            {
                "Covariate": col,
                "SMD Before": round(smd_before, 3),
                "SMD After": round(smd_after, 3),
                "Balanced (|SMD|<0.1)": abs(smd_after) < 0.1,
            }
        )
    return pd.DataFrame(rows)


# ── Figures ───────────────────────────────────────────────────────────────────


def _propensity_fig(df, ctrl_label, trt_label):
    fig = go.Figure()
    ctrl_mask = df["_T"] == 0
    fig.add_trace(
        go.Histogram(
            x=df[ctrl_mask]["_ps"],
            name=str(ctrl_label),
            opacity=0.7,
            marker_color="#7c3aed",
            nbinsx=40,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=df[~ctrl_mask]["_ps"],
            name=str(trt_label),
            opacity=0.7,
            marker_color="#db2777",
            nbinsx=40,
        )
    )
    fig.update_layout(
        title="Propensity Score Distribution",
        xaxis_title="P(treatment | covariates)",
        barmode="overlay",
        height=380,
        **_DARK,
    )
    return fig


def _balance_fig(balance: pd.DataFrame):
    if balance.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=balance["Covariate"],
            x=balance["SMD Before"].abs(),
            name="Before",
            orientation="h",
            marker_color="#64748b",
        )
    )
    fig.add_trace(
        go.Bar(
            y=balance["Covariate"],
            x=balance["SMD After"].abs(),
            name="After matching",
            orientation="h",
            marker_color="#7c3aed",
        )
    )
    fig.add_vline(
        x=0.1,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="balance threshold (0.1)",
    )
    fig.update_layout(
        title="Covariate Balance (|Standardized Mean Difference|)",
        xaxis_title="|SMD|",
        barmode="group",
        height=max(320, len(balance) * 45),
        **_DARK,
    )
    return fig


def _effect_fig(naive_diff, ate, ci, ctrl_label, trt_label, outcome_col):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[naive_diff],
            y=["Naive (unadjusted)"],
            mode="markers",
            marker=dict(size=16, color="#64748b", symbol="diamond"),
            name="Raw difference",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[ate],
            y=["ATE (adjusted)"],
            mode="markers",
            marker=dict(size=16, color="#7c3aed"),
            name="ATE",
            error_x=dict(
                type="constant",
                value=(ci[1] - ci[0]) / 2,
                color="#7c3aed",
            ),
        )
    )
    fig.add_vline(
        x=0, line_dash="dash", line_color="#94a3b8", annotation_text="no effect"
    )
    fig.update_layout(
        title=f"Treatment Effect on {outcome_col}<br>"
        f"<sup>{trt_label} vs {ctrl_label}</sup>",
        xaxis_title=f"Effect on {outcome_col}",
        height=320,
        **_DARK,
    )
    return fig
