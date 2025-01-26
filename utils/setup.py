import os
from datetime import datetime


def setup_experiment_dir(experiment_type: str) -> str:
    """
    Create and return path to timestamped experiment directory.

    Args:
        experiment_type: One of 'local_mlp', 'local_rf', 'global_mlp', 'global_rf'

    Returns:
        Path to experiment directory
    """
    # Create base results directory if it doesn't exist
    base_dir = os.path.join('results', experiment_type)
    os.makedirs(base_dir, exist_ok=True)

    # Create timestamped directory for this run
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    exp_dir = os.path.join(base_dir, timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    # Create per-problem subdirectory if needed
    if 'local' in experiment_type:
        os.makedirs(os.path.join(exp_dir, 'per_problem'), exist_ok=True)
    if 'rf' in experiment_type:
        os.makedirs(os.path.join(exp_dir, 'feature_importance'), exist_ok=True)

    return exp_dir
