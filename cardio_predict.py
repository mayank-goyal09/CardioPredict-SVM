from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC


@dataclass
class CardioPredictConfig:
    numeric_features: List[str]
    categorical_features: List[str]
    test_size: float = 0.2
    random_state: int = 42
    param_grid: Optional[Dict[str, List[Any]]] = None


class CardioPredictSVM:
    def __init__(self, config: CardioPredictConfig):
        self.config = config

        if self.config.param_grid is None:
            self.config.param_grid = {
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": [0.01, 0.1, 1, "scale"],
            }

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.config.numeric_features),
                ("cat", categorical_transformer, self.config.categorical_features),
            ]
        )

        base_svm = Pipeline(
            steps=[
                ("preprocess", self.preprocessor),
                ("model", SVC(kernel="rbf")),
            ]
        )

        self.grid_search = GridSearchCV(
            estimator=base_svm,
            param_grid=self.config.param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )

        self.best_model_ = None
        self.history_: Dict[str, Any] = {}

    def train_test_split(self, X, y):
        return train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

    def fit(self, X_train, y_train):
        self.grid_search.fit(X_train, y_train)
        self.best_model_ = self.grid_search.best_estimator_
        self.history_["best_params"] = self.grid_search.best_params_
        self.history_["best_cv_score"] = self.grid_search.best_score_
        return self

    def predict(self, X):
        return self.best_model_.predict(X)

    def decision_scores(self, X):
        return self.best_model_.decision_function(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=False)
        cm = confusion_matrix(y_test, y_pred)

        try:
            scores = self.decision_scores(X_test)
            auc = roc_auc_score(y_test, scores)
        except Exception:
            auc = None

        return {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm,
            "roc_auc": auc,
            "best_params": self.history_.get("best_params"),
            "best_cv_score": self.history_.get("best_cv_score"),
        }

    def predict_patient(self, patient_dict):
        """
        patient_dict: plain Python dict, e.g.
        {
            "Age": 55,
            "Sex": "Male",
            "BMI": 27.5,
            ...
        }
        """
        import pandas as pd

        # Build one-row DataFrame with all training columns
        X_row = pd.DataFrame([patient_dict])

        # Make sure all expected columns exist
        for col in self.config.numeric_features + self.config.categorical_features:
            if col not in X_row.columns:
                X_row[col] = np.nan

        # Order columns
        X_row = X_row[
            self.config.numeric_features + self.config.categorical_features
        ]

        pred = self.predict(X_row)[0]
        score = None
        try:
            score = float(self.decision_scores(X_row)[0])
        except Exception:
            pass

        return {"pred_class": int(pred), "score": score}


