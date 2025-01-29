from typing import Any, Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base import BaseModelTrainer


class RFTrainer(BaseModelTrainer):
    """Random Forest model trainer implementation."""

    def __init__(self, training_mode: str):
        """
        Initialize RF trainer.

        Args:
            training_mode: 'global' or 'local'
        """
        super().__init__('rf', training_mode)

        # Set different hyperparameters for global vs local
        if training_mode == 'global':
            self.n_estimators = 200
        else:
            self.n_estimators = 100

        self.max_depth = 10
        self.min_samples_split = 10
        self.min_samples_leaf = 5
        self.max_features = 'sqrt'
        self.class_weight = 'balanced'
        self.random_state = 42

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """
        Train Random Forest classifier.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained RandomForestClassifier
        """
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=-1
        )

        rf.fit(X_train, y_train)
        return rf

    def predict(self, model: RandomForestClassifier,
                X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using trained model.

        Args:
            model: Trained RandomForestClassifier
            X_test: Test features

        Returns:
            Tuple of:
                - Binary predictions
                - Prediction probabilities
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        return y_pred, y_pred_proba

    def get_feature_importance(self, model: RandomForestClassifier,
                               feature_columns: list) -> pd.DataFrame:
        """
        Get feature importance from trained model.

        Args:
            model: Trained RandomForestClassifier
            feature_columns: List of feature names

        Returns:
            DataFrame with feature importance scores
        """
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        })
        return importance_df.sort_values('importance', ascending=False)

    def get_feature_thresholds(self, model: RandomForestClassifier,
                               feature_columns: list) -> Dict[str, float]:
        """
        Get median thresholds for each feature from the RF model.

        Args:
            model: Trained RandomForestClassifier
            feature_columns: List of feature names

        Returns:
            Dictionary mapping feature names to median threshold values
        """
        feature_thresholds = {}

        for tree in model.estimators_:
            for feature_idx, threshold in zip(tree.tree_.feature, tree.tree_.threshold):
                if feature_idx >= 0:
                    feature_name = feature_columns[feature_idx]
                    if feature_name not in feature_thresholds:
                        feature_thresholds[feature_name] = []
                    feature_thresholds[feature_name].append(threshold)

        return {
            feature: np.median(thresholds)
            for feature, thresholds in feature_thresholds.items()
        }
