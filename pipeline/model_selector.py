from __future__ import annotations
import pandas as pd
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from config import CLASSIFICATION_MODELS, REGRESSION_MODELS, CV_FOLDS


def benchmark_models(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    is_ts: bool = False,
) -> tuple[pd.DataFrame, dict, str]:
    """Train and cross-validate all models for the given task.

    Returns:
        leaderboard: DataFrame ranked by primary metric
        trained_models: {name: fitted_model} for the winning model + others
        primary_metric: name of the metric used for ranking
    """
    models = (
        CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
    )
    cv = TimeSeriesSplit(n_splits=CV_FOLDS) if is_ts else CV_FOLDS
    primary_metric, scoring = _get_scoring(task_type, y)

    rows = []
    trained_models = {}
    X_arr = X.values
    y_arr = y.values

    for name, model in models.items():
        try:
            scores = cross_val_score(
                model, X_arr, y_arr, cv=cv, scoring=scoring, n_jobs=-1
            )
            mean_score = scores.mean()
            std_score = scores.std()
            # Fit on full data for later use
            model.fit(X_arr, y_arr)
            trained_models[name] = model
            rows.append(
                {
                    "model": name,
                    primary_metric: round(mean_score, 4),
                    "std": round(std_score, 4),
                }
            )
        except Exception as e:
            rows.append(
                {"model": name, primary_metric: None, "std": None, "error": str(e)}
            )

    leaderboard = pd.DataFrame(rows)
    leaderboard = leaderboard.dropna(subset=[primary_metric])
    leaderboard = leaderboard.sort_values(
        primary_metric, ascending=_lower_is_better(primary_metric)
    )
    leaderboard = leaderboard.reset_index(drop=True)

    return leaderboard, trained_models, primary_metric


def _get_scoring(task_type: str, y: pd.Series) -> tuple[str, str]:
    if task_type == "classification":
        n_classes = y.nunique()
        if n_classes == 2:
            return "AUC-ROC", "roc_auc"
        else:
            return "F1 (macro)", "f1_macro"
    else:
        return "RMSE", "neg_root_mean_squared_error"


def _lower_is_better(metric: str) -> bool:
    return "RMSE" in metric or "MAE" in metric
