"""
Model definitions, model factory, and mode-aware auxiliary classifier architecture.
"""

from typing import Dict, Any, Optional
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src.config import RANDOM_SEED

def get_baseline_models(scale_pos_weight: float = 28.0) -> Dict[str, Any]:
    """
    Returns a dictionary of candidate models configured with anti-imbalance parameters.
    """
    models = {
        "dummy": DummyClassifier(strategy="stratified", random_state=RANDOM_SEED),
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            verbose=-1,
            n_jobs=-1
        ),
    }
    return models
