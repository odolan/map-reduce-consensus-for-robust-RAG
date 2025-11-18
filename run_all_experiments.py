# run_all_experiments.py

import os
import pandas as pd

from run_experiment import run_experiment  # ensures run_experiment(...) exists


# Directory where individual experiment CSVs will be saved
RESULTS_DIR = "results"

# Experiments to run
ATTACK_TYPES = [
    "ignore_instructions",
    "biased_summary",
    "incorrect_fact",
]

SEEDS = [123, 456, 789]

TOP_K_DOCS = 10
MAX_DOCS_TO_POISON_PER_QUERY = 2


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for attack_type in ATTACK_TYPES:
        for seed in SEEDS:

            print(f"\n=== Running experiment: attack_type={attack_type}, seed={seed} ===\n")

            # Path for THIS experiment’s output CSV
            output_path = os.path.join(
                RESULTS_DIR,
                f"experiment_{attack_type}_seed{seed}.csv"
            )

            # Run one experiment with its own params
            stats_df = run_experiment(
                attack_type=attack_type,
                seed=seed,
                top_k_docs=TOP_K_DOCS,
                max_docs_to_poison_per_query=MAX_DOCS_TO_POISON_PER_QUERY,
            )

            # Save the results for this experiment
            stats_df.to_csv(output_path, index=False)
            print(f"Saved experiment results to: {output_path}")

    print("\nAll experiments complete! Individual CSVs are in the 'results/' directory.\n")


if __name__ == "__main__":
    main()
