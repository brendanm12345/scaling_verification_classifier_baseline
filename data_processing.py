from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
from datasets import load_dataset


def prepare_data(dataset_name: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare data from a given dataset for model training.

    Args:
        dataset_name: Name of the dataset to load

    Returns:
        Tuple containing:
            - DataFrame with processed data
            - List of feature column names
    """
    dataset = load_dataset(dataset_name)
    data = dataset['data']

    METADATA_COLUMNS = {
        'problem', 'samples', 'solution', 'instruction', 'type', 'level',
        'answer_correct', 'extracted_answers', 'problem_idx', 'solution_idx'
    }

    # Get first row to identify columns
    first_item = data[0]
    all_columns = set(first_item.keys())

    # Identify judge columns
    judge_columns = [col for col in all_columns
                     if col.startswith('judge_') and col not in METADATA_COLUMNS]

    # Identify numeric feature columns
    numeric_columns = [col for col in all_columns
                       if col not in METADATA_COLUMNS and
                       col not in judge_columns and
                       isinstance(first_item[col][0], (int, float, np.number))]

    # Collect verdict values and calculate modes
    verdict_values = {col: [] for col in judge_columns}
    for problem in data:
        num_samples = len(problem['samples'])
        for col in judge_columns:
            if len(problem[col]) == num_samples:
                verdict_values[col].extend(
                    [v for v in problem[col] if v is not None])

    modes = {
        col: False if not values else max(set(values), key=values.count)
        for col, values in verdict_values.items()
    }

    # Separate score columns
    score_columns = [col for col in numeric_columns
                     if col.endswith('_score') or col.endswith('_scores')]
    other_num_columns = [col for col in numeric_columns
                         if col not in score_columns]

    data_rows = []
    for idx in range(len(data)):
        problem = data[idx]
        num_samples = len(problem['samples'])

        valid_judge_columns = [
            col for col in judge_columns if len(problem[col]) == num_samples]
        valid_score_columns = [
            col for col in score_columns if len(problem[col]) == num_samples]
        valid_other_columns = [
            col for col in other_num_columns if len(problem[col]) == num_samples]

        normalized_scores = normalize_scores(problem, valid_score_columns)
        normalized_nums = normalize_scores(problem, valid_other_columns)

        for i in range(num_samples):
            if i < len(problem['answer_correct']):
                row = {
                    'problem_idx': idx,
                    'solution_idx': i,
                    'is_correct': problem['answer_correct'][i],
                    **{column: modes[column] if problem[column][i] is None
                       else problem[column][i] for column in valid_judge_columns},
                    **{column: normalized_scores[column][i] for column in valid_score_columns},
                    **{column: normalized_nums[column][i] for column in valid_other_columns}
                }
                data_rows.append(row)

    df = pd.DataFrame(data_rows)
    final_feature_columns = [col for col in (judge_columns + numeric_columns)
                             if col in df.columns]

    return df, final_feature_columns


def normalize_scores(problem: Dict, columns: List[str]) -> Dict[str, List[float]]:
    """
    Normalize scores within each problem.

    Args:
        problem: Dictionary containing problem data
        columns: List of column names to normalize

    Returns:
        Dictionary mapping column names to normalized scores
    """
    normalized = {}
    for col in columns:
        values = np.array(problem[col])
        min_val = np.min(values)
        max_val = np.max(values)
        score_range = max_val - min_val
        if score_range == 0:
            normalized[col] = [0.5] * len(values)
        else:
            normalized[col] = (values - min_val) / score_range
    return normalized
