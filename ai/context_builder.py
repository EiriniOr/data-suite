import pandas as pd
from config import CONTEXT_SAMPLE_ROWS, CONTEXT_MAX_COLS


def build_dataset_context(df: pd.DataFrame, meta: dict | None = None) -> str:
    """Serialize DataFrame statistics into a compact LLM-readable string.

    Never sends raw data — only aggregated statistics that scale to large files.
    """
    lines = []

    # ── Shape ─────────────────────────────────────────────────────────────────
    lines.append(f"**Dataset shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Column summary ────────────────────────────────────────────────────────
    cols = df.columns[:CONTEXT_MAX_COLS]
    if len(df.columns) > CONTEXT_MAX_COLS:
        lines.append(
            f"*(showing first {CONTEXT_MAX_COLS} of {len(df.columns)} columns)*"
        )

    lines.append("\n**Column overview:**")
    for col in cols:
        dtype = str(df[col].dtype)
        n_null = df[col].isna().sum()
        null_pct = n_null / len(df) * 100
        n_unique = df[col].nunique()

        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            skew = df[col].skew()
            lines.append(
                f"- `{col}` [{dtype}] — "
                f"null: {null_pct:.1f}%, unique: {n_unique}, "
                f"range: [{desc['min']:.3g}, {desc['max']:.3g}], "
                f"mean: {desc['mean']:.3g}, std: {desc['std']:.3g}, skew: {skew:.2f}"
            )
        else:
            top_vals = df[col].value_counts().head(3).index.tolist()
            lines.append(
                f"- `{col}` [{dtype}] — "
                f"null: {null_pct:.1f}%, unique: {n_unique}, "
                f"top values: {top_vals}"
            )

    # ── Null summary ──────────────────────────────────────────────────────────
    null_cols = df.isnull().sum()
    null_cols = null_cols[null_cols > 0].sort_values(ascending=False)
    if len(null_cols) > 0:
        lines.append(
            f"\n**Null columns ({len(null_cols)}):** "
            + ", ".join(f"`{c}` {v / len(df) * 100:.1f}%" for c, v in null_cols.items())
        )
    else:
        lines.append("\n**Nulls:** None")

    # ── Sample rows ───────────────────────────────────────────────────────────
    lines.append(f"\n**Sample rows (first {CONTEXT_SAMPLE_ROWS}):**")
    lines.append(df.head(CONTEXT_SAMPLE_ROWS).to_string(max_cols=CONTEXT_MAX_COLS))

    # ── Meta context (task type, target, previous decisions) ─────────────────
    if meta:
        lines.append("\n**Session context:**")
        for k, v in meta.items():
            if v is not None:
                lines.append(f"- {k}: {v}")

    return "\n".join(lines)


def build_model_results_context(leaderboard: pd.DataFrame, task_type: str) -> str:
    lines = [f"**Model benchmarking results ({task_type}):**"]
    lines.append(leaderboard.to_string(index=False))
    return "\n".join(lines)


def build_optimization_context(
    best_params: dict, baseline_score: float, best_score: float, model_name: str
) -> str:
    improvement = best_score - baseline_score
    lines = [
        f"**Optimization results for {model_name}:**",
        f"- Baseline score: {baseline_score:.4f}",
        f"- Best score after tuning: {best_score:.4f}",
        f"- Improvement: {improvement:+.4f} ({improvement / abs(baseline_score) * 100:+.1f}%)",
        "\n**Best hyperparameters:**",
    ]
    for k, v in best_params.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def build_ab_test_context(results: dict) -> str:
    lines = [
        "**A/B Test Results**",
        f"- Outcome type: {results.get('outcome_type')}",
        f"- Test used: {results.get('test_name')}",
        f"- Control: {results.get('control_label')} (n={results['sample_sizes']['control']})",
        f"- Treatment: {results.get('treatment_label')} (n={results['sample_sizes']['treatment']})",
        f"- p-value: {results.get('p_value')} (alpha={results.get('alpha', 0.05)})",
        f"- Statistically significant: {results.get('significant')}",
        f"- {results.get('effect_size_label', 'Effect size')}: {results.get('effect_size')} "
        f"({results.get('effect_interpretation')})",
        f"- {results.get('ci_label', '95% CI')}: {results.get('ci_95')}",
        f"- Achieved power: {results.get('power')}",
    ]
    if results.get("outcome_type") == "continuous":
        lines.append(
            f"- Mean difference (treatment − control): {results.get('mean_diff')}"
        )
        lines.append(
            f"- Normality assumption met: {results.get('normal_assumption_met')}"
        )
    else:
        lines.append(f"- Control rate: {results.get('control_rate', 0):.2%}")
        lines.append(f"- Treatment rate: {results.get('treatment_rate', 0):.2%}")
        lines.append(f"- Lift: {results.get('lift_pct', 0):+.1f}%")
    return "\n".join(lines)


def build_causal_context(results: dict) -> str:
    lines = [
        "**Causal Inference Results**",
        f"- Method: {results.get('method')}",
        f"- Treatment: {results.get('treatment_col')} "
        f"({results.get('treatment_label')} vs {results.get('control_label')})",
        f"- Outcome: {results.get('outcome_col')}",
        f"- n(treatment)={results.get('n_treatment')}, n(control)={results.get('n_control')}",
        f"- Naive (unadjusted) difference: {results.get('naive_diff')}",
        f"- ATE (adjusted): {results.get('ate')}",
        f"- 95% CI: {results.get('ate_ci')}",
        f"- Significant (CI excludes zero): {results.get('significant')}",
        f"- Confounder bias reduction: {results.get('bias_reduction_pct')}%",
    ]
    if results.get("matched_n"):
        lines.append(f"- Matched sample size: {results.get('matched_n')}")
    balance = results.get("balance_stats")
    if balance is not None and not balance.empty:
        n_bal = int(balance["Balanced (|SMD|<0.1)"].sum())
        lines.append(
            f"- Balance: {n_bal}/{len(balance)} covariates balanced after matching"
        )
        unbalanced = balance[~balance["Balanced (|SMD|<0.1)"]]["Covariate"].tolist()
        if unbalanced:
            lines.append(f"- Still imbalanced: {unbalanced}")
    return "\n".join(lines)
