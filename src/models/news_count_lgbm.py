import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from src.models.numeric_lgbm import LABEL_MAP, NUMERICAL_FEATURE_COLUMNS

NEWS_COUNT_FEATURE_COLUMNS = NUMERICAL_FEATURE_COLUMNS + [
    "news_coverage_count"
]

class NewsCountLGBMModel:
    """
    Model B2: Baseline that tests whether mere quantity of news predicts outcomes.
    Features: Numerical (B1) + news_coverage_count.
    """

    def __init__(self, calibrate: bool = True):
        self.feature_columns = list(NEWS_COUNT_FEATURE_COLUMNS)
        self.calibrate = calibrate
        self.model: Optional[Any] = None
        self.base_lgbm = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> "NewsCountLGBMModel":
        if y.dtype == object or isinstance(y[0], str):
            y_int = np.array([LABEL_MAP[val] for val in y], dtype=int)
        else:
            y_int = y.astype(int)

        if self.calibrate:
            calibrator = CalibratedClassifierCV(estimator=self.base_lgbm, method="sigmoid", cv=5)
            if X_val is not None and y_val is not None:
                if y_val.dtype == object or isinstance(y_val[0], str):
                    y_val_int = np.array([LABEL_MAP[val] for val in y_val], dtype=int)
                else:
                    y_val_int = y_val.astype(int)
                X_comb = np.vstack([X, X_val])
                y_comb = np.concatenate([y_int, y_val_int])
                calibrator.fit(X_comb, y_comb)
            else:
                calibrator.fit(X, y_int)
            self.model = calibrator
        else:
            self.base_lgbm.fit(X, y_int)
            self.model = self.base_lgbm

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        probs = self.model.predict_proba(X)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict_one(self, feature_dict: Dict[str, Any]) -> Dict[str, float]:
        row = np.array([[float(feature_dict.get(col, 0.0)) for col in self.feature_columns]])
        probs = self.predict_proba(row)[0]
        return {
            "home": round(float(probs[0]), 4),
            "draw": round(float(probs[1]), 4),
            "away": round(float(probs[2]), 4)
        }

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> "NewsCountLGBMModel":
        with open(filepath, "rb") as f:
            return pickle.load(f)
