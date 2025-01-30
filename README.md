## Example usage

```bash
python3 main.py --dataset "hazyresearch/CodeContests_Llama_70B_with_LM_Judges_and_RMs_v1" \
    --model-type mlp \
    --training-mode global \
    --train-splits 0.1 0.3 0.5 0.7
```

## Notes on Datasets

`hazyresearch/MATH_with_LM_Judges_and_Reward_Model_Results_V2`

- 100 problems
- 1000 generations per problem
- 27 features

`hazyresearch/AIMO_GPT-4o-mini_with_LM_Judges_and_RMs_v2`

- 90 problems
- 100 generations per problem
- 36 features

`hazyresearch/CodeContests_Llama_70B_with_LM_Judges_and_RMs_v1`

- 140 problems
- 1000 generations per problem
- 27 features

## Notes on Implementation

### Cross Validation & Hyper Paraemter Selection

After using an initial implementation of CV and HP search to choose the current hyperparamters for the models, I removed CV altogether because:

1. Letting each model choose hyperparamters as I scaled the fraction of training data did not drive significant improvements for each model configuration (local/global x rf/mlp)
2. The code and method is cleaner and more interpretable without introducing CV and HP search. Simplicity seems to be a good priority for baselines

### Hard Datasets w/ Low Positive Class Representation Are Not Well-Suited For Classifier-Based Selection

We find that when training classifiers on massively imbalanced datasets such as AIMO and CodeContests (where ~7-8% of candidate solutions are correct), they tend to achieve high generation accuracy (95-98%) simply by learning to classify most candidates as incorrect, which matches the dominant class distribution. Meanwhile, selection accuracy suffers since the model may have only seen several examples of the positive class. Strangely, we see relatively stable selection accuracy (~16-20%) across different training data sizes which suggests the model struggles to learn meaningful ranking features when correct answers are so rare - in fact, many problems have no correct answers in their training sets at all. Interestingly, we observe a relatively stable selection accuracy (~16-20%) across different training data sizes. This is because in some cases of less training data and more test data, the model encounters more test problems where all candidates are incorrect, allowing it to achieve "correct" selection simply by ranking all candidates poorly.
