import subprocess
import itertools


def get_model_configs():
    model_types = ['mlp', 'rf']
    training_modes = ['global', 'local']
    datasets = ['hazyresearch/MATH_with_LM_Judges_and_Reward_Model_Results_V2',
                'hazyresearch/AIMO_GPT-4o-mini_with_LM_Judges_and_RMs_v2', 'hazyresearch/CodeContests_Llama_70B_with_LM_Judges_and_RMs_v1']
    configs = list(itertools.product(model_types, training_modes, datasets))
    return configs


def run_training():
    configs = get_model_configs()

    for model_type, training_mode, dataset in configs:
        if training_mode == "global" and "MATH" in dataset:
            train_splits = ["0.001", "0.01", "0.1", "0.3", "0.5", "0.7"]
        else:
            train_splits = ["0.1", "0.3", "0.5", "0.7"]

        cmd = [
            "python3", "main.py",
            "--dataset", dataset,
            "--model-type", model_type,
            "--training-mode", training_mode,
            "--train-splits"] + train_splits

        print(
            f"Starting {training_mode} x {model_type} training on {dataset.replace('hazyresearch/', '')}")

        try:
            process = subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Training failed with exit code {e.returncode}")


if __name__ == "__main__":
    run_training()
