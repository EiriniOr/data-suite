from __future__ import annotations
import pickle
import io
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from utils.plot_utils import (
    confusion_matrix_chart,
    residuals_plot,
    predicted_vs_actual,
    feature_importance_chart,
)


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    is_ts: bool = False,
) -> dict:
    """Run final evaluation on a hold-out set. Returns metrics + figures."""
    X_arr, y_arr = X.values, y.values

    # Split: time-ordered for TS, random otherwise
    if is_ts:
        split = int(len(X_arr) * 0.8)
        X_train, X_test = X_arr[:split], X_arr[split:]
        y_train, y_test = y_arr[:split], y_arr[split:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr,
            y_arr,
            test_size=0.2,
            random_state=42,
            stratify=y_arr if task_type == "classification" else None,
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results: dict = {"figures": {}, "metrics": {}}

    if task_type == "classification":
        classes = sorted(np.unique(y_arr))
        cm = confusion_matrix(y_test, y_pred)
        results["figures"]["confusion_matrix"] = confusion_matrix_chart(
            cm, [str(c) for c in classes]
        )
        results["metrics"]["classification_report"] = classification_report(
            y_test, y_pred, output_dict=True
        )

        # ROC curve (binary only)
        if len(classes) == 2 and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            results["metrics"]["roc_auc"] = round(roc_auc, 4)
            results["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}

    else:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results["metrics"] = {
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4),
        }
        results["figures"]["residuals"] = residuals_plot(y_test, y_pred)
        results["figures"]["pred_vs_actual"] = predicted_vs_actual(y_test, y_pred)

    return results


def get_feature_importance(model, feature_names: list[str]) -> pd.Series | None:
    """Extract feature importances from tree-based models."""
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
        return imp.sort_values(ascending=False)
    if hasattr(model, "coef_"):
        coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        imp = pd.Series(np.abs(coef), index=feature_names)
        return imp.sort_values(ascending=False)
    return None


def pickle_model(model) -> bytes:
    """Serialize model to bytes for download."""
    buf = io.BytesIO()
    pickle.dump(model, buf)
    return buf.getvalue()


def generate_html_report(
    meta: dict,
    eval_results: dict,
    leaderboard: pd.DataFrame | None,
    ai_summary: str,
    feature_importance: pd.Series | None,
) -> str:
    """Generate a standalone HTML report with embedded charts."""
    import plotly.io as pio

    sections = []

    # Header
    sections.append(f"""
    <html><head>
    <meta charset="utf-8">
    <title>Data Suite Report — {meta.get("filename", "dataset")}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #222; }}
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;500&display=swap');
        body {{ background: #07071a; color: #e2e8f0; font-family: 'Inter', sans-serif; }}
        h1 {{ font-family: 'Space Grotesk', sans-serif; font-size: 2rem;
               background: linear-gradient(135deg,#7c3aed,#db2777);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               background-clip:text; padding-bottom: 10px; }}
        h2 {{ font-family: 'Space Grotesk', sans-serif; color: #c4b5fd; margin-top: 40px; font-size: 1.1rem;
               text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.85rem; }}
        .meta-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 20px 0; }}
        .meta-card {{ background: rgba(255,255,255,0.04); padding: 16px; border-radius: 10px;
                      border: 1px solid rgba(124,58,237,0.3); border-left: 3px solid #7c3aed; }}
        .meta-card strong {{ display: block; font-size: 0.75em; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }}
        .meta-card span {{ font-size: 1.4em; font-weight: bold; color: #e2e8f0; }}
        .ai-box {{ background: rgba(124,58,237,0.08); border: 1px solid rgba(124,58,237,0.35);
                   border-left: 3px solid #7c3aed; border-radius: 10px;
                   padding: 20px; margin: 20px 0; white-space: pre-wrap; line-height: 1.7; color: #cbd5e1; }}
        table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
        th {{ background: linear-gradient(135deg,#7c3aed,#db2777); color: white; padding: 10px 14px; text-align: left; font-family: 'Space Grotesk', sans-serif; }}
        td {{ padding: 9px 14px; border-bottom: 1px solid rgba(255,255,255,0.06); color: #94a3b8; }}
        tr:nth-child(even) {{ background: rgba(255,255,255,0.02); }}
        .footer {{ margin-top: 60px; color: #475569; font-size: 0.8em; text-align: center; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 20px; }}
    </style></head><body>
    """)

    sections.append("<h1>Data Analysis Report</h1>")
    sections.append(
        f"<p><strong>File:</strong> {meta.get('filename', 'N/A')} &nbsp;|&nbsp; "
        f"<strong>Task:</strong> {meta.get('task_type', 'N/A')} &nbsp;|&nbsp; "
        f"<strong>Target:</strong> {meta.get('target_col', 'N/A')}</p>"
    )

    # Key metrics
    sections.append('<div class="meta-grid">')
    for k, v in eval_results.get("metrics", {}).items():
        if isinstance(v, (int, float)):
            sections.append(
                f'<div class="meta-card"><strong>{k}</strong><span>{v}</span></div>'
            )
    sections.append("</div>")

    # AI summary
    if ai_summary:
        sections.append(
            f'<h2>AI Analysis Summary</h2><div class="ai-box">{ai_summary}</div>'
        )

    # Leaderboard
    if leaderboard is not None and not leaderboard.empty:
        sections.append("<h2>Model Leaderboard</h2>")
        sections.append(leaderboard.to_html(index=False, classes="", border=0))

    # Feature importance chart
    if feature_importance is not None:
        fig = feature_importance_chart(feature_importance, top_n=20)
        sections.append("<h2>Feature Importance</h2>")
        sections.append(pio.to_html(fig, full_html=False, include_plotlyjs="cdn"))

    # Eval figures
    for name, fig in eval_results.get("figures", {}).items():
        sections.append(f"<h2>{name.replace('_', ' ').title()}</h2>")
        sections.append(pio.to_html(fig, full_html=False, include_plotlyjs="cdn"))

    sections.append(
        '<div class="footer">Generated by Miss Datrix · Powered by Claude</div>'
    )
    sections.append("</body></html>")

    return "\n".join(sections)
