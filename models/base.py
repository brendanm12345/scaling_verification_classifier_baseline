from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from utils.setup import setup_experiment_dir


class BaseModelTrainer(ABC):
    """Base class for all model trainers."""

    def __init__(self, model_type: str, training_mode: str):
        """Initialize the model trainer
        Args:
            model_type: type of model ('mlp' or 'rf')
            training_mode: training mode ('local' or 'global')
        """
        self.model_type = model_type
        self.training_mode = training_mode
        self.exp_dir = setup_experiment_dir(f"{training_mode}_{model_type}")

    @abstractmethod
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Train model"""
        pass

    @abstractmethod
    def predict(self, model: Any, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make prediction using trained model"""
        pass

    def calculate_metrics(self, df_test: pd.DataFrame,
                          y_pred: np.ndarray,
                          y_pred_proba: np.ndarray,
                          model: Optional[Any] = None,
                          feature_columns: Optional[List[str]] = None) -> Dict:
        """Calcuate metrics for model predictions
        Args:
            df_test: test data
            y_pred: binary predictions
            y_pred_proba: prediction probabilities
            model: optional training model (to get feature importance etc.)
            feature_columns: optional list of feature names

        Returns:
            Dictionary of metrics
        """
        if self.training_mode == 'local':
            return self._calculate_local_metrics(df_test, y_pred, y_pred_proba)
        else:
            return self._calculate_global_metrics(df_test, y_pred, y_pred_proba)

    def _calculate_global_metrics(self, df_test: pd.DataFrame,
                                  y_pred: np.ndarray,
                                  y_pred_proba: np.ndarray,
                                  model: Optional[Any] = None,
                                  feature_columns: Optional[List[str]] = None) -> Dict:
        """Calculate metrics for global model."""
        metrics_per_problem = []
        total_correct_predictions = 0
        total_predictions = 0

        for prob_idx in df_test['problem_idx'].unique():
            prob_mask = df_test['problem_idx'] == prob_idx
            prob_df = df_test[prob_mask]
            prob_proba = y_pred_proba[prob_mask]
            prob_pred = y_pred[prob_mask]

            correct_predictions = (prob_pred == prob_df['is_correct']).sum()
            total_predictions += len(prob_pred)
            total_correct_predictions += correct_predictions

            selected_idx = np.argmax(prob_proba)
            prob_labels = prob_df['is_correct'].values

            metrics_per_problem.append(self._calculate_selection_metrics(
                prob_labels, selected_idx))

        return self._aggregate_metrics(metrics_per_problem,
                                       total_correct_predictions,
                                       total_predictions)

    def _calculate_local_metrics(self, df_test: pd.DataFrame,
                                 y_pred: np.ndarray,
                                 y_pred_proba: np.ndarray) -> Dict:
        """Calculate metrics for local model (single problem)."""
        generation_accuracy = (y_pred == df_test['is_correct']).mean()

        selected_idx = np.argmax(y_pred_proba)
        prob_labels = df_test['is_correct'].values

        metrics = self._calculate_selection_metrics(prob_labels, selected_idx)
        metrics.update({
            'problem_idx': df_test['problem_idx'].iloc[0],
            'generation_accuracy': generation_accuracy
        })

        return metrics

    @staticmethod
    def _calculate_selection_metrics(prob_labels: np.ndarray,
                                     selected_idx: int) -> Dict:
        """Calculate selection metrics for a single problem."""
        tp = 1 if prob_labels[selected_idx] else 0
        fp = 1 if not prob_labels[selected_idx] else 0
        fn = 1 if sum(prob_labels) > 0 and not prob_labels[selected_idx] else 0
        tn = 1 if sum(
            prob_labels) == 0 and not prob_labels[selected_idx] else 0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return {
            'selection_accuracy': tp,
            'selection_precision': precision,
            'selection_recall': recall,
            'selection_f1': 2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0 else 0,
            'selection_tp': tp,
            'selection_tn': tn,
            'selection_fp': fp,
            'selection_fn': fn
        }

    @staticmethod
    def _aggregate_metrics(metrics_per_problem: List[Dict],
                           total_correct: int,
                           total_predictions: int) -> Dict:
        """Aggregate metrics across problems."""
        n_problems = len(metrics_per_problem)

        aggregated = {
            'generation_accuracy': total_correct / total_predictions,
            'selection_accuracy': np.mean([m['selection_accuracy'] for m in metrics_per_problem]),
            'selection_precision': np.mean([m['selection_precision'] for m in metrics_per_problem]),
            'selection_recall': np.mean([m['selection_recall'] for m in metrics_per_problem]),
            'selection_f1': np.mean([m['selection_f1'] for m in metrics_per_problem]),
            'selection_tp': sum(m['selection_tp'] for m in metrics_per_problem),
            'selection_tn': sum(m['selection_tn'] for m in metrics_per_problem),
            'selection_fp': sum(m['selection_fp'] for m in metrics_per_problem),
            'selection_fn': sum(m['selection_fn'] for m in metrics_per_problem)
        }

        return aggregated
