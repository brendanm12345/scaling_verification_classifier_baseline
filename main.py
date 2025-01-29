import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict

from data_processing import prepare_data
from models.mlp import MLPTrainer
from models.rf import RFTrainer
from training.strategies import train_global_model, train_local_models


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and evaluate baseline models')

    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to use (e.g., "hazyresearch/CodeContests_Llama_70B_with_LM_Judges_and_RMs_v1")')

    parser.add_argument('--model-type', type=str, required=True,
                        choices=['mlp', 'rf'],
                        help='Type of model to train')

    parser.add_argument('--training-mode', type=str, required=True,
                        choices=['global', 'local'],
                        help='Training mode (global or local)')

    parser.add_argument('--train-splits', type=float, nargs='+',
                        default=[0.001, 0.01, 0.1, 0.3, 0.5, 0.7],
                        help='List of training split percentages')

    return parser.parse_args()


def plot_results(results_df: pd.DataFrame, model_type: str,
                 training_mode: str, exp_dir: str):
    """Plot accuracy metrics."""
    plt.figure(figsize=(10, 5))

    # Selection@1 plot
    plt.subplot(1, 2, 1)
    plt.plot(results_df['train_percentage'] * 100,
             results_df['selection_accuracy'],
             '-o', label=f'{training_mode.title()} {model_type.upper()}')
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Selection@1')
    plt.title('Selection@1 vs Training Data Size')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.legend()

    # Generation Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(results_df['train_percentage'] * 100,
             results_df['generation_accuracy'],
             '-o', label=f'{training_mode.title()} {model_type.upper()}')
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Generation Accuracy')
    plt.title('Generation Accuracy vs Training Data Size')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'plots.png'))
    plt.close()


def main():
    args = parse_args()

    # Prepare data
    print(f"Preparing data from {args.dataset}...")
    df, feature_columns = prepare_data(args.dataset)

    # Init trainer
    if args.model_type == 'mlp':
        trainer = MLPTrainer(args.training_mode)
    else:
        trainer = RFTrainer(args.training_mode)

    # Train models for each split
    results = []
    feature_importances = []  # only for RF

    for train_percentage in args.train_splits:
        print(f"\nTraining with {train_percentage*100}% of data")

        if args.training_mode == 'global':
            metrics = train_global_model(
                trainer, df, feature_columns, train_percentage)

            # Save feature importance for RF
            if args.model_type == 'rf':
                importance_df = trainer.get_feature_importance(
                    metrics['model'], feature_columns)
                feature_importances.append({
                    'train_percentage': train_percentage,
                    'importance_df': importance_df
                })
        else:
            metrics = train_local_models(
                trainer, df, feature_columns, train_percentage)

        metrics['train_percentage'] = train_percentage
        results.append(metrics)

        print(f"Selection accuracy: {metrics['selection_accuracy']:.3f}")
        print(f"Generation accuracy: {metrics['generation_accuracy']:.3f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(
        trainer.exp_dir, 'metrics.csv'), index=False)

    if args.model_type == 'rf' and args.training_mode == 'global':
        importance_dir = os.path.join(trainer.exp_dir, 'feature_importance')
        os.makedirs(importance_dir, exist_ok=True)
        for result in feature_importances:
            result['importance_df'].to_csv(
                os.path.join(importance_dir, f'feature_importance_{
                             result["train_percentage"]}.csv'),
                index=False
            )

    # Plot results
    plot_results(results_df, args.model_type,
                 args.training_mode, trainer.exp_dir)


if __name__ == '__main__':
    main()
