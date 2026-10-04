"""
Model definitions, model factory, and mode-aware auxiliary classifier architecture.

Implements standard candidate models and the innovative ModeAwareStackingClassifier,
which trains auxiliary classifiers on failure modes (HDF, PWF, OSF, TWF) using strictly
out-of-fold predictions during training to eliminate in-sample target leakage.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import RANDOM_SEED
from src.utils import logger


def get_baseline_models(scale_pos_weight: float = 28.0) -> Dict[str, Any]:
    """
    Returns candidate models configured with anti-imbalance hyperparameters.
    Imbalance ratio is ~28.5 (96.6% vs 3.4%).
    """
    return {
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


class ModeAwareStackingClassifier(BaseEstimator, ClassifierMixin):
    """
    Mode-Aware Auxiliary Stacking Architecture.
    
    Learns auxiliary models for individual failure modes:
        - Model_HDF: Heat Dissipation Failure
        - Model_PWF: Power Failure
        - Model_OSF: Overstrain Failure
        - Model_TWF: Tool Wear Failure
        
    CRITICAL ANTI-LEAKAGE DESIGN:
    1. During training, auxiliary probabilities for training samples are generated
       STRICTLY OUT-OF-FOLD using internal K-Fold cross-validation. No sample is ever
       predicted by an auxiliary estimator that was trained on that sample.
    2. At inference time, the failure mode labels (HDF, PWF, OSF, TWF) DO NOT EXIST.
       The refit auxiliary models generate predicted probabilities purely from standard
       sensor features.
    3. The main model receives either:
       - 'stacking': Base sensor features + auxiliary probabilities
       - 'noisy_or': Analytical probability combination P = 1 - Prod(1 - P_mode)
    """

    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        auxiliary_estimator: Optional[Any] = None,
        combination_mode: str = "stacking",  # 'stacking' or 'noisy_or'
        n_inner_folds: int = 5,
        random_state: int = RANDOM_SEED,
    ):
        self.base_estimator = base_estimator
        self.auxiliary_estimator = auxiliary_estimator
        self.combination_mode = combination_mode
        self.n_inner_folds = n_inner_folds
        self.random_state = random_state
        
        # State populated during fit
        self.modes_ = ["HDF", "PWF", "OSF", "TWF"]
        self.aux_models_: Dict[str, Any] = {}
        self.main_model_: Any = None
        self.classes_: np.ndarray = np.array([0, 1])
        self.oof_train_predictions_: Optional[pd.DataFrame] = None
        self.training_indices_used_: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        aux_targets: Optional[pd.DataFrame] = None,
    ):
        """
        Fits auxiliary mode models with strict internal out-of-fold cross-validation,
        then fits the main classifier.
        """
        X_df = pd.DataFrame(X).copy().reset_index(drop=True)
        y_arr = np.asarray(y)
        n_samples = len(X_df)

        if aux_targets is None or not all(m in aux_targets.columns for m in self.modes_):
            logger.warning(
                "ModeAwareClassifier fit called without full auxiliary targets %s. "
                "Falling back to standard single model.", self.modes_
            )
            self.main_model_ = (
                clone(self.base_estimator)
                if self.base_estimator is not None
                else LGBMClassifier(class_weight="balanced", random_state=self.random_state, verbose=-1)
            )
            self.main_model_.fit(X_df, y_arr)
            return self

        aux_df = aux_targets[self.modes_].copy().reset_index(drop=True)

        # ----------------------------------------------------------------------
        # STEP 1: Strict Out-Of-Fold Auxiliary Feature Generation
        # Generate auxiliary probability features for X using internal K-Fold.
        # ----------------------------------------------------------------------
        oof_aux = pd.DataFrame(0.0, index=np.arange(n_samples), columns=[f"prob_{m}" for m in self.modes_])
        inner_cv = StratifiedKFold(n_splits=self.n_inner_folds, shuffle=True, random_state=self.random_state)
        
        # Track indices for automated anti-leakage audit
        self.training_indices_used_ = []

        for fold_idx, (in_train_idx, in_val_idx) in enumerate(inner_cv.split(X_df, y_arr)):
            self.training_indices_used_.append((in_train_idx, in_val_idx))
            X_in_train, X_in_val = X_df.iloc[in_train_idx], X_df.iloc[in_val_idx]

            for mode in self.modes_:
                y_mode_train = aux_df[mode].iloc[in_train_idx]
                
                # Check if positive mode instances exist in inner fold
                if len(np.unique(y_mode_train)) > 1:
                    aux_clf = (
                        clone(self.auxiliary_estimator)
                        if self.auxiliary_estimator is not None
                        else LGBMClassifier(n_estimators=50, random_state=self.random_state + fold_idx, verbose=-1, n_jobs=-1)
                    )
                    aux_clf.fit(X_in_train, y_mode_train)
                    probs_val = aux_clf.predict_proba(X_in_val)[:, 1]
                else:
                    probs_val = np.zeros(len(in_val_idx))

                # Store strictly in the validation indices (zero in-sample leakage)
                oof_aux.loc[in_val_idx, f"prob_{mode}"] = probs_val

        self.oof_train_predictions_ = oof_aux.copy()

        # ----------------------------------------------------------------------
        # STEP 2: Refit Final Auxiliary Models on Full Training Set
        # These final auxiliary models will predict on completely unseen test inputs.
        # ----------------------------------------------------------------------
        self.aux_models_ = {}
        for mode in self.modes_:
            y_mode = aux_df[mode]
            if len(np.unique(y_mode)) > 1:
                aux_clf = (
                    clone(self.auxiliary_estimator)
                    if self.auxiliary_estimator is not None
                    else LGBMClassifier(n_estimators=50, random_state=self.random_state, verbose=-1, n_jobs=-1)
                )
                aux_clf.fit(X_df, y_mode)
                self.aux_models_[mode] = aux_clf
            else:
                self.aux_models_[mode] = None

        # ----------------------------------------------------------------------
        # STEP 3: Fit Main Model
        # ----------------------------------------------------------------------
        if self.combination_mode == "stacking":
            X_augmented = pd.concat([X_df, oof_aux], axis=1)
            self.main_model_ = (
                clone(self.base_estimator)
                if self.base_estimator is not None
                else LGBMClassifier(class_weight="balanced", random_state=self.random_state, verbose=-1)
            )
            self.main_model_.fit(X_augmented, y_arr)
        elif self.combination_mode == "noisy_or":
            # In noisy-OR mode, the failure probability is modeled analytically
            # No additional main model needed, or fit a 1D calibrator
            self.main_model_ = None
        else:
            raise ValueError(f"Unknown combination_mode: '{self.combination_mode}'")

        return self

    def _predict_auxiliary_probs(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Generates auxiliary failure mode probabilities from raw/transformed sensor features."""
        X_df = pd.DataFrame(X).copy().reset_index(drop=True)
        aux_probs = pd.DataFrame(index=X_df.index)
        
        for mode in self.modes_:
            clf = self.aux_models_.get(mode)
            if clf is not None:
                aux_probs[f"prob_{mode}"] = clf.predict_proba(X_df)[:, 1]
            else:
                aux_probs[f"prob_{mode}"] = 0.0
                
        return aux_probs

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Inference prediction. Computes auxiliary probabilities on the fly from input features,
        then feeds them into the main model or computes noisy-OR combination.
        """
        X_df = pd.DataFrame(X).copy().reset_index(drop=True)
        
        if not self.aux_models_:
            return self.main_model_.predict_proba(X_df)

        aux_probs = self._predict_auxiliary_probs(X_df)

        if self.combination_mode == "stacking":
            X_augmented = pd.concat([X_df, aux_probs], axis=1)
            return self.main_model_.predict_proba(X_augmented)
        elif self.combination_mode == "noisy_or":
            # Noisy-OR combination: P(failure) = 1 - Prod(1 - P(mode_i))
            survival_prob = np.prod(1.0 - aux_probs.values, axis=1)
            failure_prob = np.clip(1.0 - survival_prob, 0.0, 1.0)
            return np.column_stack([1.0 - failure_prob, failure_prob])
        else:
            raise ValueError(f"Unknown combination_mode: {self.combination_mode}")

    def predict(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
