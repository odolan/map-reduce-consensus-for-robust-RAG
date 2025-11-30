# run_all_experiments to automate multiple runs with different hyperparameters

import os
import pandas as pd

from run_experiment import run_experiment

RESULTS_DIR = "results"

# attack experiment types
ATTACK_TYPES = [
    "ignore_instructions",
    "biased_summary",
    "incorrect_fact",
]

# seeds
SEEDS = [123]

# hyperparameter search space
TOP_K_DOCS_LIST = [5, 10]
MAX_DOCS_TO_POISON_PER_QUERY_LIST = [1, 4, 8]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # iterate through all hyperparams combos
    for attack_type in ATTACK_TYPES:
        for top_k_docs in TOP_K_DOCS_LIST:
            for max_docs_to_poison in MAX_DOCS_TO_POISON_PER_QUERY_LIST:

                # name this results folder attack_top_k_max_poison
                exp_name = f"attack={attack_type}_topk={top_k_docs}_maxpoison={max_docs_to_poison}"

                # create new results folder
                exp_dir = os.path.join(RESULTS_DIR, exp_name)
                os.makedirs(exp_dir, exist_ok=True)

                print(f"\nRunning experiments for {exp_name}\n")

                all_stats_dfs = []

                for seed in SEEDS:
                    print(f" -> seed={seed}")

                    # run one experiment for this seed and hyperparam combo
                    stats_df = run_experiment(
                        attack_type=attack_type,
                        seed=seed,
                        top_k_docs=top_k_docs,
                        max_docs_to_poison_per_query=max_docs_to_poison,
                    )

                    # add the seed to the df so we can track later if needed
                    stats_df["seed"] = seed
                    all_stats_dfs.append(stats_df)

                # combine across seeds and make one hyperparam level csv
                if all_stats_dfs:
                    combined_df = pd.concat(all_stats_dfs, ignore_index=True)

                    # one CSV per attack, top_k, max_poison
                    stats_path = os.path.join(exp_dir, "stats.csv")
                    combined_df.to_csv(stats_path, index=False)
                    print(f"Saved aggregated stats (over seeds) to: {stats_path}")


    print("\nCompleted all experiments. Saved to results directory.")


if __name__ == "__main__":
    main()