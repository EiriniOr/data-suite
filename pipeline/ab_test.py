"""A/B test analysis — statistical significance, effect size, power."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def detect_outcome_type(series: pd.Series) -> str:
    n_unique = series.nunique()
    if n_unique == 2:
        return "binary"
    if series.dtype == object and n_unique <= 5:
        return "binary"
    return "continuous"


def run_ab_test(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    alpha: float = 0.05,
) -> dict:
    """
    Run A/B test analysis.

    Returns metrics, group stats, and plotly figures.
    outcome_type is auto-detected: "continuous" → t-test / Mann-Whitney,
    "binary" → chi-square.
    """
    groups = df[treatment_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError(
            f"Treatment column must have exactly 2 groups, found: {sorted(str(g) for g in groups)}"
        )

    groups_sorted = sorted(groups, key=str)
    ctrl_label, trt_label = groups_sorted[0], groups_sorted[1]

    ctrl = df[df[treatment_col] == ctrl_label][outcome_col].dropna()
    trt = df[df[treatment_col] == trt_label][outcome_col].dropna()

    outcome_type = detect_outcome_type(df[outcome_col])
    results: dict = {
        "outcome_type": outcome_type,
        "control_label": str(ctrl_label),
        "treatment_label": str(trt_label),
        "sample_sizes": {"control": len(ctrl), "treatment": len(trt)},
        "alpha": alpha,
        "figures": {},
    }

    if outcome_type == "continuous":
        _continuous_test(
            results,
            ctrl,
            trt,
            ctrl_label,
            trt_label,
            outcome_col,
            df,
            treatment_col,
            alpha,
        )
    else:
        _binary_test(
            results,
            ctrl,
            trt,
            ctrl_label,
            trt_label,
            outcome_col,
            df,
            treatment_col,
            alpha,
        )

    return results


# ── Continuous outcome ────────────────────────────────────────────────────────


def _continuous_test(
    results, ctrl, trt, ctrl_label, trt_label, outcome_col, df, treatment_col, alpha
):
    # Normality check (Shapiro on sample ≤ 50)
    _, p_c = stats.shapiro(ctrl.sample(min(50, len(ctrl)), random_state=42))
    _, p_t = stats.shapiro(trt.sample(min(50, len(trt)), random_state=42))
    normal = p_c > 0.05 and p_t > 0.05

    if normal:
        stat, p_value = stats.ttest_ind(ctrl, trt, equal_var=False)
        test_name = "Welch's t-test"
    else:
        stat, p_value = stats.mannwhitneyu(ctrl, trt, alternative="two-sided")
        test_name = "Mann-Whitney U"

    # Cohen's d
    pooled_std = np.sqrt(
        ((len(ctrl) - 1) * ctrl.std() ** 2 + (len(trt) - 1) * trt.std() ** 2)
        / (len(ctrl) + len(trt) - 2)
    )
    cohens_d = (trt.mean() - ctrl.mean()) / pooled_std if pooled_std > 0 else 0.0

    # 95% CI for mean difference
    diff = trt.mean() - ctrl.mean()
    se = np.sqrt(ctrl.var() / len(ctrl) + trt.var() / len(trt))
    ci = (diff - 1.96 * se, diff + 1.96 * se)

    # Post-hoc power
    nc = abs(cohens_d) * np.sqrt(len(ctrl) * len(trt) / (len(ctrl) + len(trt)))
    power = float(1 - stats.norm.cdf(stats.norm.ppf(1 - alpha / 2) - nc))

    group_stats = pd.DataFrame(
        {
            "Group": [str(ctrl_label), str(trt_label)],
            "N": [len(ctrl), len(trt)],
            "Mean": [round(ctrl.mean(), 4), round(trt.mean(), 4)],
            "Std": [round(ctrl.std(), 4), round(trt.std(), 4)],
            "Median": [round(float(ctrl.median()), 4), round(float(trt.median()), 4)],
        }
    )

    results.update(
        {
            "test_name": test_name,
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 6),
            "effect_size": round(float(cohens_d), 4),
            "effect_size_label": "Cohen's d",
            "effect_interpretation": _interpret_cohens_d(cohens_d),
            "ci_95": (round(float(ci[0]), 4), round(float(ci[1]), 4)),
            "ci_label": "95% CI (mean difference)",
            "mean_diff": round(float(diff), 4),
            "power": round(power, 4),
            "group_stats": group_stats,
            "significant": bool(p_value < alpha),
            "normal_assumption_met": normal,
        }
    )

    results["figures"]["distribution"] = _distribution_fig(
        ctrl, trt, str(ctrl_label), str(trt_label), outcome_col
    )
    results["figures"]["violin"] = _violin_fig(df, treatment_col, outcome_col)


# ── Binary outcome ────────────────────────────────────────────────────────────


def _binary_test(
    results, ctrl, trt, ctrl_label, trt_label, outcome_col, df, treatment_col, alpha
):
    contingency = pd.crosstab(df[treatment_col], df[outcome_col])
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

    n = len(df)
    cramers_v = float(np.sqrt(chi2 / (n * (min(contingency.shape) - 1))))

    # Conversion rates — treat larger unique value as "positive"
    pos_val = sorted(df[outcome_col].dropna().unique(), key=str)[-1]
    ctrl_rate = (ctrl == pos_val).mean()
    trt_rate = (trt == pos_val).mean()
    diff = trt_rate - ctrl_rate
    lift = diff / ctrl_rate * 100 if ctrl_rate > 0 else 0.0

    se = np.sqrt(
        ctrl_rate * (1 - ctrl_rate) / len(ctrl) + trt_rate * (1 - trt_rate) / len(trt)
    )
    ci = (diff - 1.96 * se, diff + 1.96 * se)

    # Power
    pooled = (ctrl_rate * len(ctrl) + trt_rate * len(trt)) / (len(ctrl) + len(trt))
    se_null = np.sqrt(pooled * (1 - pooled) * (1 / len(ctrl) + 1 / len(trt)))
    z = diff / se_null if se_null > 0 else 0.0
    power = float(stats.norm.cdf(abs(z) - stats.norm.ppf(1 - alpha / 2)))

    group_stats = pd.DataFrame(
        {
            "Group": [str(ctrl_label), str(trt_label)],
            "N": [len(ctrl), len(trt)],
            "Rate": [f"{ctrl_rate:.2%}", f"{trt_rate:.2%}"],
            "Count (positive)": [
                int((ctrl == pos_val).sum()),
                int((trt == pos_val).sum()),
            ],
        }
    )

    results.update(
        {
            "test_name": "Chi-square test",
            "statistic": round(float(chi2), 4),
            "p_value": round(float(p_value), 6),
            "effect_size": round(cramers_v, 4),
            "effect_size_label": "Cramér's V",
            "effect_interpretation": _interpret_cramers_v(cramers_v),
            "ci_95": (round(float(ci[0]), 4), round(float(ci[1]), 4)),
            "ci_label": "95% CI (rate difference)",
            "control_rate": ctrl_rate,
            "treatment_rate": trt_rate,
            "rate_diff": round(float(diff), 4),
            "lift_pct": round(float(lift), 2),
            "power": round(power, 4),
            "group_stats": group_stats,
            "significant": bool(p_value < alpha),
        }
    )

    results["figures"]["bar"] = _conversion_bar_fig(
        ctrl_rate, trt_rate, str(ctrl_label), str(trt_label), outcome_col
    )
    results["figures"]["contingency"] = _contingency_heatmap(contingency)


# ── Interpretation helpers ────────────────────────────────────────────────────


def _interpret_cohens_d(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def _interpret_cramers_v(v: float) -> str:
    if v < 0.1:
        return "negligible"
    if v < 0.3:
        return "small"
    if v < 0.5:
        return "medium"
    return "large"


# ── Figures ───────────────────────────────────────────────────────────────────

_DARK = {
    "template": "plotly_dark",
    "paper_bgcolor": "#07071a",
    "plot_bgcolor": "#0f0f2d",
}


def _distribution_fig(ctrl, trt, ctrl_label, trt_label, col_name):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Histogram", "Box"])
    fig.add_trace(
        go.Histogram(
            x=ctrl, name=ctrl_label, opacity=0.7, marker_color="#7c3aed", nbinsx=30
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=trt, name=trt_label, opacity=0.7, marker_color="#db2777", nbinsx=30
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Box(y=ctrl, name=ctrl_label, marker_color="#7c3aed"), row=1, col=2)
    fig.add_trace(go.Box(y=trt, name=trt_label, marker_color="#db2777"), row=1, col=2)
    fig.update_layout(
        title=f"{col_name} — Group Comparison",
        barmode="overlay",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **_DARK,
    )
    return fig


def _violin_fig(df, treatment_col, outcome_col):
    fig = px.violin(
        df,
        x=treatment_col,
        y=outcome_col,
        color=treatment_col,
        box=True,
        points="outliers",
        color_discrete_sequence=["#7c3aed", "#db2777"],
        title=f"{outcome_col} by {treatment_col}",
    )
    fig.update_layout(height=420, **_DARK)
    return fig


def _conversion_bar_fig(ctrl_rate, trt_rate, ctrl_label, trt_label, outcome_col):
    fig = go.Figure(
        [
            go.Bar(
                x=[ctrl_label],
                y=[ctrl_rate],
                name=ctrl_label,
                marker_color="#7c3aed",
                text=[f"{ctrl_rate:.2%}"],
                textposition="outside",
            ),
            go.Bar(
                x=[trt_label],
                y=[trt_rate],
                name=trt_label,
                marker_color="#db2777",
                text=[f"{trt_rate:.2%}"],
                textposition="outside",
            ),
        ]
    )
    fig.update_layout(
        title=f"Conversion Rate: {outcome_col}",
        yaxis_tickformat=".0%",
        height=400,
        **_DARK,
    )
    return fig


def _contingency_heatmap(contingency: pd.DataFrame):
    fig = px.imshow(
        contingency,
        text_auto=True,
        color_continuous_scale=["#0f0f2d", "#7c3aed"],
        title="Contingency Table",
    )
    fig.update_layout(height=350, **_DARK)
    return fig
