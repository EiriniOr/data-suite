import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


PALETTE = px.colors.qualitative.Set2


def null_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of missing values across columns."""
    null_matrix = df.isnull().astype(int)
    fig = px.imshow(
        null_matrix.T,
        color_continuous_scale=["#e8f5e9", "#e53935"],
        labels={"color": "Is Null"},
        title="Missing Value Map (red = null)",
        aspect="auto",
    )
    fig.update_coloraxes(showscale=False)
    return fig


def distribution_plot(series: pd.Series) -> go.Figure:
    """Histogram + KDE for a numeric series."""
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(
        go.Histogram(
            x=series.dropna(),
            name="Count",
            marker_color=PALETTE[0],
            opacity=0.75,
            nbinsx=40,
        )
    )
    fig.update_layout(title=f"Distribution: {series.name}", bargap=0.05)
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Pearson correlation heatmap for numeric columns."""
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap",
        text_auto=".2f",
    )
    fig.update_layout(height=600)
    return fig


def class_balance_chart(series: pd.Series) -> go.Figure:
    """Bar chart of target class distribution."""
    counts = series.value_counts().reset_index()
    counts.columns = ["class", "count"]
    fig = px.bar(
        counts,
        x="class",
        y="count",
        title=f"Class Distribution: {series.name}",
        color="class",
        color_discrete_sequence=PALETTE,
    )
    return fig


def time_series_plot(df: pd.DataFrame, dt_col: str, value_col: str) -> go.Figure:
    """Line chart for a time series column."""
    fig = px.line(
        df,
        x=dt_col,
        y=value_col,
        title=f"{value_col} over time",
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(line_width=1.5)
    return fig


def model_leaderboard_chart(leaderboard: pd.DataFrame, metric: str) -> go.Figure:
    """Horizontal bar chart ranking models by metric."""
    df_sorted = leaderboard.sort_values(metric)
    fig = px.bar(
        df_sorted,
        x=metric,
        y="model",
        orientation="h",
        title=f"Model Comparison — {metric}",
        color=metric,
        color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis_title="", showlegend=False)
    return fig


def confusion_matrix_chart(cm: np.ndarray, labels: list) -> go.Figure:
    """Plotly heatmap of a confusion matrix."""
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        text_auto=True,
    )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    return fig


def residuals_plot(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Residuals vs predicted for regression."""
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode="markers",
            marker=dict(color=PALETTE[0], opacity=0.5),
            name="Residuals",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title="Residuals vs Predicted", xaxis_title="Predicted", yaxis_title="Residual"
    )
    return fig


def predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Scatter of predicted vs actual for regression."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker=dict(color=PALETTE[1], opacity=0.5),
            name="Predictions",
        )
    )
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Perfect fit",
        )
    )
    fig.update_layout(
        title="Predicted vs Actual", xaxis_title="Actual", yaxis_title="Predicted"
    )
    return fig


def feature_importance_chart(importances: pd.Series, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of top N feature importances."""
    top = importances.nlargest(top_n).sort_values()
    fig = px.bar(
        x=top.values,
        y=top.index,
        orientation="h",
        title=f"Top {top_n} Feature Importances",
        color=top.values,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis_title="", showlegend=False)
    return fig


def optimization_history_chart(trials_df: pd.DataFrame, metric: str) -> go.Figure:
    """Line chart showing best score improvement over Optuna trials."""
    trials_df = trials_df.copy()
    trials_df["best_so_far"] = trials_df[metric].cummax()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trials_df["number"],
            y=trials_df[metric],
            mode="markers",
            marker=dict(color=PALETTE[2], opacity=0.4),
            name="Trial score",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trials_df["number"],
            y=trials_df["best_so_far"],
            mode="lines",
            line=dict(color=PALETTE[0], width=2),
            name="Best so far",
        )
    )
    fig.update_layout(
        title="Optimization Progress", xaxis_title="Trial", yaxis_title=metric
    )
    return fig
