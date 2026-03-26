from __future__ import annotations
import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from config import OPTUNA_TRIALS, CV_FOLDS

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize(
    model_name: str,
    task_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    is_ts: bool = False,
    n_trials: int = OPTUNA_TRIALS,
    progress_callback=None,
) -> tuple[object, dict, float, float, pd.DataFrame]:
    """Run Optuna hyperparameter search for the selected model.

    Returns:
        best_model: fitted model with best params
        best_params: dict of hyperparameters
        baseline_score: CV score before tuning (default params)
        best_score: CV score after tuning
        trials_df: DataFrame of all trials for plotting
    """
    cv = TimeSeriesSplit(n_splits=CV_FOLDS) if is_ts else CV_FOLDS
    _, scoring = _get_scoring(task_type, y)
    X_arr, y_arr = X.values, y.values

    # Baseline score (default model params)
    default_model = _build_model(model_name, task_type, {})
    baseline_scores = cross_val_score(
        default_model, X_arr, y_arr, cv=cv, scoring=scoring, n_jobs=-1
    )
    baseline_score = baseline_scores.mean()

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_name, task_type)
        model = _build_model(model_name, task_type, params)
        scores = cross_val_score(model, X_arr, y_arr, cv=cv, scoring=scoring, n_jobs=-1)
        score = scores.mean()
        if progress_callback:
            progress_callback(trial.number + 1, score)
        return score

    direction = "minimize" if _lower_is_better(scoring) else "maximize"
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value

    # Fit final model on full data
    best_model = _build_model(model_name, task_type, best_params)
    best_model.fit(X_arr, y_arr)

    # Build trials DataFrame
    trials_df = study.trials_dataframe()[["number", "value"]].rename(
        columns={"value": scoring}
    )

    return best_model, best_params, baseline_score, best_score, trials_df


def _get_scoring(task_type: str, y: pd.Series | None = None) -> tuple[str, str]:
    if task_type == "classification":
        return "AUC-ROC", "roc_auc"
    return "RMSE", "neg_root_mean_squared_error"


def _lower_is_better(scoring: str) -> bool:
    return "neg_" in scoring


def _build_model(model_name: str, task_type: str, params: dict):
    is_clf = task_type == "classification"
    builders = {
        "LogisticRegression": lambda p: LogisticRegression(
            max_iter=1000, random_state=42, **p
        ),
        "Ridge": lambda p: Ridge(**p),
        "RandomForest": lambda p: (
            RandomForestClassifier(random_state=42, **p)
            if is_clf
            else RandomForestRegressor(random_state=42, **p)
        ),
        "XGBoost": lambda p: (
            XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0, **p)
            if is_clf
            else XGBRegressor(random_state=42, verbosity=0, **p)
        ),
        "LightGBM": lambda p: (
            LGBMClassifier(random_state=42, verbose=-1, **p)
            if is_clf
            else LGBMRegressor(random_state=42, verbose=-1, **p)
        ),
    }
    return builders[model_name](params)


def _suggest_params(trial: optuna.Trial, model_name: str, task_type: str) -> dict:
    if model_name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    elif model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    elif model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
    elif model_name == "LogisticRegression":
        return {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
        }
    elif model_name == "Ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True)}
    return {}
