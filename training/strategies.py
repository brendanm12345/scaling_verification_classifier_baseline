from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from joblib import Parallel, delayed

from models.base import BaseModelTrainer


def train_global_model(trainer: BaseModelTrainer,
                       df: pd.DataFrame,
                       feature_columns: List[str],
                       train_percentage: float) -> Dict:
    """Train a global model using specified percentage of data per problem.

    Args:
        trainer: model trainer instance
        df: full dataset
        feature_columns: list of feature column names
        train_percentage: fraction of data to use for training
    Returns:
        Dictionary of metrics
    """
    X = df[feature_columns].copy()
    y = df['is_correct']

    problems_per_generation = len(df) / len(df['problem_idx'].unique())
    k = max(1, int(problems_per_generation * train_percentage))
    train_mask = df.groupby('problem_idx').cumcount() < k

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[~train_mask]
    df_test = df[~train_mask]

    model = trainer.train_model(X_train, y_train)
    y_train_pred, y_train_proba = trainer.predict(model, X_train)
    train_metrics = trainer.calculate_metrics(
        df[train_mask], y_train_pred, y_train_proba)

    # Calculate metrics on test set
    y_test_pred, y_test_proba = trainer.predict(model, X_test)
    test_metrics = trainer.calculate_metrics(
        df_test, y_test_pred, y_test_proba)

    # Combine metrics
    metrics = {
        'train_' + k: v for k, v in train_metrics.items()
    }
    metrics.update(test_metrics)

    metrics['train_percentage'] = train_percentage
    metrics['model'] = model

    return metrics


def train_local_model(trainer: BaseModelTrainer,
                      problem_data: Tuple[pd.DataFrame, float, List[str]]) -> Dict:
    """
    Train a model for a single problem.

    Args:
        trainer: Model trainer instance
        problem_data: Tuple containing:
            - DataFrame for single problem
            - Training percentage
            - Feature column names

    Returns:
        Dictionary of metrics
    """
    df_prob, train_percentage, feature_columns = problem_data

    X = df_prob[feature_columns].copy()
    y = df_prob['is_correct']

    n_positive = y.sum()

    # Handle insufficient data case
    if n_positive <= 3:
        majority_class = int(y.mean() >= 0.5)
        y_pred = np.full(len(y), majority_class)
        y_pred_proba = np.full(len(y), float(majority_class))
        metrics = trainer.calculate_metrics(df_prob, y_pred, y_pred_proba)
        metrics['insufficient_data'] = True
        return metrics

    # Train model on sufficient data (n_splits cannot be lower than 2)
    skf = StratifiedKFold(n_splits=max(2, int(np.ceil(1/train_percentage))),
                          shuffle=True, random_state=42)
    train_idx, test_idx = next(skf.split(X, y))

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    df_test = df_prob.iloc[test_idx]

    model = trainer.train_model(X_train, y_train)
    y_pred, y_pred_proba = trainer.predict(model, X_test)

    metrics = trainer.calculate_metrics(df_test, y_pred, y_pred_proba)
    metrics['insufficient_data'] = False
    return metrics


def train_local_models(trainer: BaseModelTrainer,
                       df: pd.DataFrame,
                       feature_columns: List[str],
                       train_percentage: float) -> Dict:
    """
    Train local models for all problems in parallel.

    Args:
        trainer: Model trainer instance
        df: Full dataset
        feature_columns: List of feature column names
        train_percentage: Fraction of data to use for training

    Returns:
        Dictionary of aggregated metrics
    """
    problem_groups = [
        (group, train_percentage, feature_columns)
        for _, group in df.groupby('problem_idx')
    ]

    problem_metrics = Parallel(n_jobs=-1, verbose=1)(
        delayed(train_local_model)(trainer, problem_data)
        for problem_data in problem_groups
    )

    return aggregate_local_metrics(problem_metrics)


def aggregate_local_metrics(problem_metrics: List[Dict]) -> Dict:
    """
    Aggregate metrics across all local models.

    Args:
        problem_metrics: List of metric dictionaries from individual problems

    Returns:
        Dictionary of aggregated metrics
    """
    return {
        'generation_accuracy': np.mean([m['generation_accuracy'] for m in problem_metrics]),
        'selection_accuracy': np.mean([m['selection_accuracy'] for m in problem_metrics]),
        'selection_precision': np.mean([m['selection_precision'] for m in problem_metrics]),
        'selection_recall': np.mean([m['selection_recall'] for m in problem_metrics]),
        'selection_f1': np.mean([m['selection_f1'] for m in problem_metrics]),
        'selection_tp': sum(m['selection_tp'] for m in problem_metrics),
        'selection_tn': sum(m['selection_tn'] for m in problem_metrics),
        'selection_fp': sum(m['selection_fp'] for m in problem_metrics),
        'selection_fn': sum(m['selection_fn'] for m in problem_metrics),
        'total_problems': len(problem_metrics),
        'insufficient_data_problems': sum(1 for m in problem_metrics if m.get('insufficient_data', False))
    }
