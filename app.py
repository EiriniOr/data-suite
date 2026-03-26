"""Data Suite — AI-powered end-to-end data analysis platform."""

from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

from pipeline.loader import load_file, infer_task_type, DatasetMeta
from pipeline.cleaner import clean, auto_null_strategies, outlier_summary
from pipeline.eda import (
    compute_summary,
    get_distribution_figures,
    get_correlation_figure,
    get_target_distribution,
    get_ts_figures,
    flag_issues,
)
from pipeline.feature_eng import engineer_features
from pipeline.model_selector import benchmark_models
from pipeline.optimizer import optimize
from pipeline.reporter import (
    evaluate_model,
    get_feature_importance,
    pickle_model,
    generate_html_report,
)
from ai.analyst import DataAnalyst
from utils.plot_utils import (
    null_heatmap,
    model_leaderboard_chart,
    optimization_history_chart,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state init ────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "stage": "upload",
        "df_raw": None,
        "df_clean": None,
        "df_engineered": None,
        "X": None,
        "y": None,
        "meta": None,
        "analyst": DataAnalyst(),
        "ai_response": {},  # {stage: response text}
        "leaderboard": None,
        "trained_models": None,
        "primary_metric": None,
        "best_model": None,
        "best_model_name": None,
        "best_params": None,
        "baseline_score": None,
        "best_score": None,
        "trials_df": None,
        "eval_results": None,
        "feature_importance": None,
        "workflow": [],
        "cleaning_strategies": None,
        "outlier_action": "clip",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Data Suite")
    st.caption("AI-guided data analysis")

    # Why not Claude.ai?
    with st.expander("Why not just use Claude?", expanded=False):
        st.markdown("""
**Claude.ai can *talk* about your data.**
This app *runs* against it.

✓ Loads your real file — no copy-pasting
✓ Trains actual models you can download
✓ Runs Optuna (50+ tuning trials)
✓ Computes SHAP / feature importance
✓ Interactive Plotly charts
✓ Exports HTML report + `.pkl` model
✓ AI scales to large files (reads stats, not raw rows)
✓ Workflow adapts to your goal
        """)

    st.divider()

    # Stage indicator
    stages = ["upload", "clean", "eda", "features", "models", "optimize", "report"]
    stage_labels = {
        "upload": "📂 Load & Intent",
        "clean": "🧹 Clean",
        "eda": "🔍 EDA",
        "features": "⚙️ Features",
        "models": "🤖 Models",
        "optimize": "🎯 Optimize",
        "report": "📊 Report",
    }
    current = st.session_state.stage
    for s in stages:
        if s not in st.session_state.workflow and s != "upload":
            continue
        label = stage_labels[s]
        if s == current:
            st.markdown(f"**→ {label}**")
        elif stages.index(s) < stages.index(current):
            st.markdown(f"~~{label}~~ ✓")
        else:
            st.markdown(f"  {label}")

    st.divider()

    # Persistent chat
    if st.session_state.df_raw is not None:
        st.subheader("💬 Ask the Analyst")
        user_q = st.text_input(
            "Ask anything about your data",
            key="chat_input",
            label_visibility="collapsed",
            placeholder="e.g. Why was XGBoost chosen?",
        )
        if user_q:
            with st.spinner("Thinking..."):
                answer = st.session_state.analyst.chat(user_q)
            st.markdown(answer)


# ── Helpers ───────────────────────────────────────────────────────────────────
def ai_panel(stage: str, response_key: str | None = None):
    """Show AI response for a stage. Cache in session state."""
    key = response_key or stage
    if key in st.session_state.ai_response:
        with st.container(border=True):
            st.markdown("**🤖 AI Analyst**")
            st.markdown(st.session_state.ai_response[key])


def go_to(stage: str):
    st.session_state.stage = stage
    st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 0: UPLOAD + INTENT
# ════════════════════════════════════════════════════════════════════════════════
if st.session_state.stage == "upload":
    st.title("Data Suite")
    st.subheader("Upload your dataset and tell the AI what you want to do")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader(
            "Choose a file (CSV, Excel, Parquet)",
            type=["csv", "xlsx", "xls", "parquet"],
        )
        user_goal = st.text_area(
            "What do you want to do? *(optional)*",
            placeholder='e.g. "predict customer churn", "just explore the data", "I don\'t know yet"',
            height=80,
        )

    with col2:
        st.info("**Supported formats**\n- CSV\n- Excel (.xlsx / .xls)\n- Parquet")

    if uploaded and st.button("🚀 Start Analysis", type="primary"):
        with st.spinner("Loading file..."):
            try:
                df, meta = load_file(uploaded)
                st.session_state.df_raw = df
                st.session_state.meta = meta
            except Exception as e:
                st.error(f"Failed to load file: {e}")
                st.stop()

        with st.spinner("AI is analyzing your dataset..."):
            try:
                response = st.session_state.analyst.analyze_intent(df, user_goal)
                st.session_state.ai_response["intent"] = response
            except Exception as e:
                st.warning(f"AI analysis unavailable (check ANTHROPIC_API_KEY): {e}")
                response = ""

        # Show data preview
        st.subheader("Dataset Preview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing values", f"{df.isnull().sum().sum():,}")
        st.dataframe(df.head(10), use_container_width=True)

        # Show null map
        if df.isnull().sum().sum() > 0:
            st.plotly_chart(null_heatmap(df), use_container_width=True)

        # AI response
        if response:
            with st.container(border=True):
                st.markdown("**🤖 AI Analyst says:**")
                st.markdown(response)
            st.text_area(
                "Your response / answers to AI questions (optional)",
                key="intent_reply",
                height=80,
            )

        # Target + task type
        st.subheader("Configure Task")
        all_cols = df.columns.tolist()
        target_col = st.selectbox(
            "Select target column", all_cols, index=len(all_cols) - 1
        )
        meta.target_col = target_col

        inferred_task = infer_task_type(df, target_col)
        if meta.is_ts:
            inferred_task = "time_series"
        task_type = st.selectbox(
            "Task type",
            ["classification", "regression", "time_series"],
            index=["classification", "regression", "time_series"].index(inferred_task),
        )
        meta.task_type = task_type

        # If user replied to AI, send it
        if st.session_state.get("intent_reply"):
            with st.spinner("AI is updating..."):
                try:
                    follow_up = st.session_state.analyst.chat(
                        st.session_state.intent_reply
                    )
                    st.session_state.ai_response["intent_followup"] = follow_up
                except Exception:
                    pass

        # Workflow selection
        st.subheader("Workflow")
        full_workflow = ["clean", "eda", "features", "models", "optimize", "report"]
        selected_workflow = st.multiselect(
            "Select stages to run",
            full_workflow,
            default=full_workflow,
            format_func=lambda s: stage_labels[s],
        )

        if st.button("✅ Confirm & Proceed", type="primary"):
            meta.approved_workflow = selected_workflow
            st.session_state.workflow = ["upload"] + selected_workflow
            next_stage = selected_workflow[0] if selected_workflow else "report"
            go_to(next_stage)


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 1: CLEAN
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "clean":
    st.title("🧹 Data Cleaning")
    df = st.session_state.df_raw
    meta: DatasetMeta = st.session_state.meta

    # AI analysis
    if "clean" not in st.session_state.ai_response:
        with st.spinner("AI is reviewing data quality..."):
            try:
                response = st.session_state.analyst.analyze_cleaning(df, meta.to_dict())
                st.session_state.ai_response["clean"] = response
            except Exception as e:
                st.session_state.ai_response["clean"] = f"*(AI unavailable: {e})*"
    ai_panel("clean")

    st.divider()

    # Auto strategies (user can override)
    auto_strategies = auto_null_strategies(df)
    st.subheader("Null Imputation Strategy")
    null_cols = {c: df[c].isna().sum() for c in df.columns if df[c].isna().sum() > 0}

    if null_cols:
        strategies = {}
        for col, n in null_cols.items():
            pct = n / len(df) * 100
            default = auto_strategies.get(col, "keep")
            strat = st.selectbox(
                f"`{col}` — {n} nulls ({pct:.1f}%)",
                ["median", "mode", "mean", "knn", "zero", "drop_col", "keep"],
                index=[
                    "median",
                    "mode",
                    "mean",
                    "knn",
                    "zero",
                    "drop_col",
                    "keep",
                ].index(default),
                key=f"null_{col}",
            )
            strategies[col] = strat
    else:
        st.success("No null values found!")
        strategies = {}

    # Outlier handling
    st.subheader("Outlier Handling")
    out_summary = outlier_summary(df)
    if not out_summary.empty and out_summary["n_outliers"].sum() > 0:
        st.dataframe(
            out_summary[out_summary["n_outliers"] > 0], use_container_width=True
        )
    outlier_action = st.radio(
        "Outlier action", ["clip", "drop", "keep"], horizontal=True
    )

    # Duplicates
    n_dupes = df.duplicated().sum()
    drop_dupes = st.checkbox(f"Remove {n_dupes} duplicate rows", value=n_dupes > 0)

    if st.button("🧹 Apply Cleaning", type="primary"):
        with st.spinner("Cleaning..."):
            df_clean, report = clean(df, strategies, outlier_action, drop_dupes)
        st.session_state.df_clean = df_clean
        meta.cleaning_report = report

        # Show report
        st.success(
            f"Cleaned: {report['duplicates_removed']} dupes removed, "
            f"{len(report['dropped_cols'])} cols dropped, "
            f"{len(report['imputed'])} cols imputed, "
            f"{len(report['outliers'])} cols with outliers handled."
        )
        st.subheader("Before / After")
        c1, c2 = st.columns(2)
        c1.metric("Rows before", len(df))
        c2.metric("Rows after", len(df_clean))
        st.dataframe(df_clean.head(), use_container_width=True)

        workflow = st.session_state.meta.approved_workflow
        idx = workflow.index("clean") if "clean" in workflow else -1
        next_stage = workflow[idx + 1] if idx + 1 < len(workflow) else "report"
        if st.button("Next →", key="next_clean"):
            go_to(next_stage)


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 2: EDA
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "eda":
    st.title("🔍 Exploratory Data Analysis")
    df = st.session_state.df_clean or st.session_state.df_raw
    meta: DatasetMeta = st.session_state.meta

    if "eda" not in st.session_state.ai_response:
        with st.spinner("AI is analyzing patterns..."):
            try:
                response = st.session_state.analyst.analyze_eda(df, meta.to_dict())
                st.session_state.ai_response["eda"] = response
            except Exception as e:
                st.session_state.ai_response["eda"] = f"*(AI unavailable: {e})*"
    ai_panel("eda")

    st.divider()
    summary = compute_summary(df)

    # Warnings
    issues = flag_issues(df, meta.target_col, meta.task_type)
    if issues:
        with st.expander("⚠️ Data Quality Flags", expanded=True):
            for w in issues:
                st.warning(w)

    # Numeric summary table
    st.subheader("Numeric Statistics")
    st.dataframe(
        summary["numeric_stats"].style.background_gradient(
            cmap="Blues", subset=["mean", "std"]
        ),
        use_container_width=True,
    )

    # Target distribution
    if meta.target_col and meta.target_col in df.columns:
        st.subheader("Target Distribution")
        st.plotly_chart(
            get_target_distribution(df, meta.target_col, meta.task_type),
            use_container_width=True,
        )

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    st.plotly_chart(get_correlation_figure(df), use_container_width=True)

    # Distributions
    st.subheader("Column Distributions")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if meta.target_col in numeric_cols:
        numeric_cols.remove(meta.target_col)
    selected_cols = st.multiselect(
        "Select columns", numeric_cols, default=numeric_cols[:4]
    )
    if selected_cols:
        figs = get_distribution_figures(df, selected_cols)
        cols = st.columns(min(2, len(selected_cols)))
        for i, (col, fig) in enumerate(figs.items()):
            cols[i % 2].plotly_chart(fig, use_container_width=True)

    # Time series plots
    if meta.is_ts and meta.datetime_col:
        st.subheader("Time Series")
        ts_num_cols = df.select_dtypes(include="number").columns.tolist()[:4]
        ts_figs = get_ts_figures(df, meta.datetime_col, ts_num_cols)
        for col, fig in ts_figs.items():
            st.plotly_chart(fig, use_container_width=True)

    workflow = st.session_state.meta.approved_workflow
    idx = workflow.index("eda") if "eda" in workflow else -1
    next_stage = workflow[idx + 1] if idx + 1 < len(workflow) else "report"
    if st.button("Next →", key="next_eda"):
        go_to(next_stage)


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 3: FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "features":
    st.title("⚙️ Feature Engineering")
    df = st.session_state.df_clean or st.session_state.df_raw
    meta: DatasetMeta = st.session_state.meta

    if "features" not in st.session_state.ai_response:
        with st.spinner("AI is planning features..."):
            try:
                response = st.session_state.analyst.analyze_features(df, meta.to_dict())
                st.session_state.ai_response["features"] = response
            except Exception as e:
                st.session_state.ai_response["features"] = f"*(AI unavailable: {e})*"
    ai_panel("features")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        scaler_type = st.selectbox(
            "Numeric scaling",
            ["standard", "minmax", "robust", "none"],
            help="Standard: zero-mean unit-variance. Robust: IQR-based (outlier resistant). None: for tree models.",
        )
    with c2:
        drop_high_corr = st.checkbox(
            "Drop highly correlated features (r > 0.95)", value=True
        )

    lag_windows, rolling_windows = None, None
    if meta.is_ts:
        st.subheader("Time Series Features")
        lag_input = st.text_input("Lag windows (comma-separated)", "1,7,30")
        roll_input = st.text_input("Rolling windows (comma-separated)", "7,30")
        try:
            lag_windows = [int(x.strip()) for x in lag_input.split(",") if x.strip()]
            rolling_windows = [
                int(x.strip()) for x in roll_input.split(",") if x.strip()
            ]
        except ValueError:
            st.error("Enter valid integers separated by commas.")

    if st.button("⚙️ Engineer Features", type="primary"):
        with st.spinner("Engineering features..."):
            try:
                X, y, feat_report = engineer_features(
                    df,
                    target_col=meta.target_col,
                    task_type=meta.task_type,
                    scaler_type=scaler_type,
                    drop_high_corr=drop_high_corr,
                    is_ts=meta.is_ts,
                    dt_col=meta.datetime_col,
                    lag_windows=lag_windows,
                    rolling_windows=rolling_windows,
                )
                st.session_state.X = X
                st.session_state.y = y
                meta.feature_report = feat_report
            except Exception as e:
                st.error(f"Feature engineering failed: {e}")
                st.stop()

        st.success(f"Features ready: {X.shape[1]} features × {len(X):,} rows")
        c1, c2, c3 = st.columns(3)
        c1.metric("Features added", len(feat_report.get("added", [])))
        c2.metric("Columns encoded", len(feat_report.get("encoded", {})))
        c3.metric("Columns dropped", len(feat_report.get("dropped", [])))
        st.dataframe(X.head(), use_container_width=True)

        workflow = st.session_state.meta.approved_workflow
        idx = workflow.index("features") if "features" in workflow else -1
        next_stage = workflow[idx + 1] if idx + 1 < len(workflow) else "report"
        if st.button("Next →", key="next_features"):
            go_to(next_stage)


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 4: MODEL SELECTION
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "models":
    st.title("🤖 Model Selection")
    X = st.session_state.X
    y = st.session_state.y
    meta: DatasetMeta = st.session_state.meta

    if X is None:
        st.error("Run Feature Engineering first.")
        st.stop()

    if st.session_state.leaderboard is None:
        if st.button("🚀 Benchmark Models", type="primary"):
            progress = st.progress(0, text="Training models...")
            with st.spinner("Benchmarking..."):
                leaderboard, trained_models, primary_metric = benchmark_models(
                    X,
                    y,
                    meta.task_type,
                    meta.is_ts,
                )
            progress.progress(100)
            st.session_state.leaderboard = leaderboard
            st.session_state.trained_models = trained_models
            st.session_state.primary_metric = primary_metric
            st.rerun()
    else:
        leaderboard = st.session_state.leaderboard
        primary_metric = st.session_state.primary_metric

        # AI analysis
        if "models" not in st.session_state.ai_response:
            with st.spinner("AI is interpreting results..."):
                try:
                    response = st.session_state.analyst.analyze_models(
                        leaderboard, meta.to_dict()
                    )
                    st.session_state.ai_response["models"] = response
                except Exception as e:
                    st.session_state.ai_response["models"] = f"*(AI unavailable: {e})*"
        ai_panel("models")

        st.divider()
        st.subheader("Leaderboard")
        st.dataframe(
            leaderboard.style.highlight_max(subset=[primary_metric], color="#d4edda"),
            use_container_width=True,
        )
        st.plotly_chart(
            model_leaderboard_chart(leaderboard, primary_metric),
            use_container_width=True,
        )

        best_name = leaderboard.iloc[0]["model"]
        selected_model_name = st.selectbox(
            "Select model for optimization",
            leaderboard["model"].tolist(),
            index=0,
            help=f"Auto-selected: {best_name} (top scorer)",
        )
        st.session_state.best_model_name = selected_model_name
        meta.best_model_name = selected_model_name
        meta.best_model_score = leaderboard.iloc[0][primary_metric]
        meta.best_model_metric = primary_metric

        workflow = st.session_state.meta.approved_workflow
        idx = workflow.index("models") if "models" in workflow else -1
        next_stage = workflow[idx + 1] if idx + 1 < len(workflow) else "report"
        if st.button("Next →", key="next_models"):
            go_to(next_stage)


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 5: OPTIMIZE
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "optimize":
    st.title("🎯 Hyperparameter Optimization")
    X = st.session_state.X
    y = st.session_state.y
    meta: DatasetMeta = st.session_state.meta
    model_name = st.session_state.best_model_name or "XGBoost"

    if X is None:
        st.error("Run Feature Engineering first.")
        st.stop()

    n_trials = st.slider("Number of Optuna trials", 10, 200, 50)
    st.caption(
        "More trials = better params but longer runtime. 50 is a good starting point."
    )

    if st.session_state.best_model is None:
        if st.button(f"🎯 Optimize {model_name}", type="primary"):
            progress_bar = st.progress(0, text="Running Optuna trials...")

            def update_progress(trial_num, score):
                pct = int(trial_num / n_trials * 100)
                progress_bar.progress(
                    pct, text=f"Trial {trial_num}/{n_trials} — best score so far"
                )

            with st.spinner("Optimizing..."):
                best_model, best_params, baseline, best_score, trials_df = optimize(
                    model_name,
                    meta.task_type,
                    X,
                    y,
                    is_ts=meta.is_ts,
                    n_trials=n_trials,
                    progress_callback=update_progress,
                )
            progress_bar.progress(100)

            st.session_state.best_model = best_model
            st.session_state.best_params = best_params
            st.session_state.baseline_score = baseline
            st.session_state.best_score = best_score
            st.session_state.trials_df = trials_df
            st.rerun()
    else:
        best_params = st.session_state.best_params
        baseline = st.session_state.baseline_score
        best_score = st.session_state.best_score
        trials_df = st.session_state.trials_df
        primary_metric = st.session_state.primary_metric or "score"

        # AI analysis
        if "optimize" not in st.session_state.ai_response:
            with st.spinner("AI is interpreting results..."):
                try:
                    response = st.session_state.analyst.analyze_optimization(
                        best_params, baseline, best_score, model_name
                    )
                    st.session_state.ai_response["optimize"] = response
                except Exception as e:
                    st.session_state.ai_response["optimize"] = (
                        f"*(AI unavailable: {e})*"
                    )
        ai_panel("optimize")

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline score", f"{baseline:.4f}")
        c2.metric("Best score", f"{best_score:.4f}")
        c3.metric("Improvement", f"{best_score - baseline:+.4f}")

        st.subheader("Best Hyperparameters")
        st.json(best_params)

        st.subheader("Optimization History")
        st.plotly_chart(
            optimization_history_chart(trials_df, trials_df.columns[-1]),
            use_container_width=True,
        )

        workflow = st.session_state.meta.approved_workflow
        idx = workflow.index("optimize") if "optimize" in workflow else -1
        next_stage = workflow[idx + 1] if idx + 1 < len(workflow) else "report"
        if st.button("Next →", key="next_opt"):
            go_to(next_stage)


# ════════════════════════════════════════════════════════════════════════════════
# STAGE 6: REPORT
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "report":
    st.title("📊 Results & Report")
    X = st.session_state.X
    y = st.session_state.y
    meta: DatasetMeta = st.session_state.meta
    model = st.session_state.best_model or (
        list(st.session_state.trained_models.values())[0]
        if st.session_state.trained_models
        else None
    )

    if model is None or X is None:
        st.warning("No trained model found. Complete earlier stages first.")
        st.stop()

    # Evaluate if not done
    if st.session_state.eval_results is None:
        with st.spinner("Running final evaluation..."):
            eval_results = evaluate_model(model, X, y, meta.task_type, meta.is_ts)
            feature_imp = get_feature_importance(model, X.columns.tolist())
            st.session_state.eval_results = eval_results
            st.session_state.feature_importance = feature_imp

    eval_results = st.session_state.eval_results
    feature_imp = st.session_state.feature_importance

    # AI summary
    if "report" not in st.session_state.ai_response:
        with st.spinner("AI is writing summary..."):
            try:
                summary = st.session_state.analyst.generate_report_summary(
                    meta.to_dict()
                )
                st.session_state.ai_response["report"] = summary
            except Exception as e:
                st.session_state.ai_response["report"] = f"*(AI unavailable: {e})*"
    ai_panel("report")

    st.divider()

    # Metrics
    st.subheader("Performance Metrics")
    metrics = eval_results.get("metrics", {})
    if isinstance(metrics, dict):
        display_metrics = {
            k: v for k, v in metrics.items() if isinstance(v, (int, float))
        }
        if display_metrics:
            cols = st.columns(len(display_metrics))
            for i, (k, v) in enumerate(display_metrics.items()):
                cols[i].metric(k, f"{v:.4f}")

    # Feature importance
    if feature_imp is not None:
        st.subheader("Feature Importance")
        from utils.plot_utils import feature_importance_chart

        st.plotly_chart(
            feature_importance_chart(feature_imp, top_n=20), use_container_width=True
        )

    # Eval figures
    for name, fig in eval_results.get("figures", {}).items():
        st.subheader(name.replace("_", " ").title())
        st.plotly_chart(fig, use_container_width=True)

    # Downloads
    st.divider()
    st.subheader("Downloads")
    dcol1, dcol2 = st.columns(2)

    with dcol1:
        model_bytes = pickle_model(model)
        st.download_button(
            "⬇️ Download Trained Model (.pkl)",
            data=model_bytes,
            file_name=f"{meta.best_model_name or 'model'}.pkl",
            mime="application/octet-stream",
        )

    with dcol2:
        with st.spinner("Generating HTML report..."):
            html = generate_html_report(
                meta=meta.to_dict(),
                eval_results=eval_results,
                leaderboard=st.session_state.leaderboard,
                ai_summary=st.session_state.ai_response.get("report", ""),
                feature_importance=feature_imp,
            )
        st.download_button(
            "⬇️ Download HTML Report",
            data=html,
            file_name="data_suite_report.html",
            mime="text/html",
        )

    if st.button("🔄 Start New Analysis"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
