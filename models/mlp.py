from typing import Any, Tuple
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

from models.base import BaseModelTrainer


class MLPTrainer(BaseModelTrainer):
    """
    Initialize MLP trainer.

    Args:
        training_mode: 'global' or 'local'
        dataset: name of dataset
    """

    def __init__(self, training_mode: str, dataset: str):
        super().__init__('mlp', training_mode, dataset)

        if training_mode == 'global':
            self.hidden_layer_sizes = (100, 50)
        else:
            self.hidden_layer_sizes = (100,)

        self.learning_rate = 0.001
        self.max_iter = 1000
        self.random_state = 42

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> MLPClassifier:
        """Train MLP Classifier
            Args:
                X_train: training features
                y_train: training labels
            Returns:
                A trained MLP
        """
        mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            random_state=self.random_state
        )

        mlp.fit(X_train, y_train)
        return mlp

    def predict(self, model: MLPClassifier,
                X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using trained model.

        Args:
            model: trained MLPClassifier
            X_test: test features

        Returns:
            Binary predictions & prediction probabilities
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        return y_pred, y_pred_proba
