from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# ── Cleaning ──────────────────────────────────────────────────────────────────
NULL_DROP_THRESHOLD = 0.5  # drop column if >50% null
OUTLIER_IQR_FACTOR = 1.5  # IQR multiplier for outlier detection
HIGH_CARDINALITY_THRESHOLD = 20  # use target encoding above this many unique values

# ── Model benchmarking ────────────────────────────────────────────────────────
CV_FOLDS = 5

CLASSIFICATION_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    "SVM": SVC(probability=True, random_state=42),
}

REGRESSION_MODELS = {
    "Ridge": Ridge(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(random_state=42, verbosity=0),
    "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
    "SVR": SVR(),
}

# ── Optuna tuning ─────────────────────────────────────────────────────────────
OPTUNA_TRIALS = 50

# ── AI ────────────────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-6"
CONTEXT_SAMPLE_ROWS = 5
CONTEXT_MAX_COLS = 30  # truncate context if dataset has more columns
