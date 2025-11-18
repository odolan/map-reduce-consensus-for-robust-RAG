import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# CONFIG – EDIT THESE
# ==========================
CSV_FILES = [
    "results/incorrect_fact_results.csv",
    "results/ignore_instructions_results.csv",
    "results/biased_summary_results.csv",
]
QUERIES_JSONL_PATH = "data/queries.jsonl"   # path to your queries jsonl
OUTPUT_DIR = "rag_attack_plots"
# ==========================


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_and_tag(csv_files):
    """Load each CSV and add an 'attack_type' column based on the filename."""
    dfs = []
    for path in csv_files:
        df = pd.read_csv(path)

        # infer attack type from filename (strip folder + extension)
        attack_type = os.path.splitext(os.path.basename(path))[0]
        df["attack_type"] = attack_type

        # cast booleans (in case they came in as strings)
        bool_cols = [
            "consensus_agent_used_poisoned_doc",
            "consensus_agent_output_was_poisoned",
            "default_agent_used_poisoned_doc",
            "default_agent_output_was_poisoned",
        ]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        # ensure query_id is string to match jsonl
        if "query_id" in df.columns:
            df["query_id"] = df["query_id"].astype(str)

        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_queries_jsonl(path: str):
    """
    Load queries jsonl into a dict:
      query_id -> {
          "query_text": str,
          "ground_truth": str,
          "doc_ids": set([...])
      }
    """
    queries = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = str(obj.get("query_id"))
            queries[qid] = {
                "query_text": obj.get("query_text", ""),
                "ground_truth": obj.get("ground_truth", ""),
                "doc_ids": set(obj.get("doc_ids", [])),
            }
    return queries


# ---------- Combined Charts (all attacks) ----------

def plot_attack_success_comparison(all_df: pd.DataFrame, outdir: str):
    """
    For each attack_type, compare:
      - % of queries where default output was poisoned
      - % of queries where consensus output was poisoned
    """
    grouped = all_df.groupby("attack_type")
    attack_types = list(grouped.groups.keys())

    default_rates = []
    consensus_rates = []
    for atk, g in grouped:
        default_rates.append(g["default_agent_output_was_poisoned"].mean() * 100.0)
        consensus_rates.append(g["consensus_agent_output_was_poisoned"].mean() * 100.0)

    x = range(len(attack_types))
    width = 0.35

    plt.figure()
    plt.bar([i - width / 2 for i in x], default_rates, width=width, label="Default RAG")
    plt.bar([i + width / 2 for i in x], consensus_rates, width=width, label="Consensus RAG")

    plt.xticks(x, attack_types, rotation=20, ha="right")
    plt.ylabel("Poisoned output rate (%)")
    plt.title("Attack Success Rate: Default vs Consensus")
    for i, v in enumerate(default_rates):
        plt.text(i - width / 2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(consensus_rates):
        plt.text(i + width / 2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    plt.ylim(0, max(default_rates + consensus_rates + [10]) * 1.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "attack_success_default_vs_consensus.png"))
    plt.close()


def plot_poison_stage_survival(all_df: pd.DataFrame, outdir: str):
    """
    For each attack_type + method (default/consensus), show probabilities:
      Stage 1: poison retrieved
      Stage 2: poison used in reasoning
      Stage 3: poisoned output
    """
    grouped = all_df.groupby("attack_type")
    attack_types = list(grouped.groups.keys())

    stages_default = {"retrieved": [], "used": [], "output": []}
    stages_cons = {"retrieved": [], "used": [], "output": []}

    for atk, g in grouped:
        # default
        retrieved_default = (g["default_num_selected_relevant_poisoned"] > 0).mean() * 100.0
        used_default = g["default_agent_used_poisoned_doc"].mean() * 100.0
        output_default = g["default_agent_output_was_poisoned"].mean() * 100.0

        stages_default["retrieved"].append(retrieved_default)
        stages_default["used"].append(used_default)
        stages_default["output"].append(output_default)

        # consensus
        retrieved_cons = (g["num_selected_relevant_poisoned"] > 0).mean() * 100.0
        used_cons = g["consensus_agent_used_poisoned_doc"].mean() * 100.0
        output_cons = g["consensus_agent_output_was_poisoned"].mean() * 100.0

        stages_cons["retrieved"].append(retrieved_cons)
        stages_cons["used"].append(used_cons)
        stages_cons["output"].append(output_cons)

    x = range(len(attack_types))
    width = 0.12

    plt.figure(figsize=(10, 5))
    offsets = {
        ("Default", "retrieved"): -3 * width,
        ("Default", "used"): -width,
        ("Default", "output"): width,
        ("Consensus", "retrieved"): -2 * width,
        ("Consensus", "used"): 0,
        ("Consensus", "output"): 2 * width,
    }

    def bar_vals(method_stages, method_name):
        for stage_name, vals in method_stages.items():
            x_positions = [i + offsets[(method_name, stage_name)] for i in x]
            plt.bar(x_positions, vals, width=width, label=f"{method_name} – {stage_name}", alpha=0.8)

    bar_vals(stages_default, "Default")
    bar_vals(stages_cons, "Consensus")

    plt.xticks(x, attack_types, rotation=20, ha="right")
    plt.ylabel("Probability (%)")
    plt.title("Poison Survival Through Pipeline Stages (Default vs Consensus)")
    plt.ylim(0, 110)
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys(), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "poison_stage_survival.png"))
    plt.close()


def plot_consensus_drop_behavior(all_df: pd.DataFrame, outdir: str):
    """
    For each attack_type, compute average drop rates for:
      - poisoned relevant docs
      - clean relevant docs
    (consensus pipeline only)
    """
    grouped = all_df.groupby("attack_type")
    attack_types = list(grouped.groups.keys())

    poison_drop_rates = []
    clean_drop_rates = []

    for atk, g in grouped:
        # Poison drop rate: num_dropped_relevant_poisoned / num_selected_relevant_poisoned
        mask_poison_selected = g["num_selected_relevant_poisoned"] > 0
        if mask_poison_selected.any():
            pr = (
                g.loc[mask_poison_selected, "num_dropped_relevant_poisoned"]
                / g.loc[mask_poison_selected, "num_selected_relevant_poisoned"]
            ).mean()
        else:
            pr = 0.0

        # Clean drop rate: num_dropped_relevant_clean / num_selected_relevant_clean
        mask_clean_selected = g["num_selected_relevant_clean"] > 0
        if mask_clean_selected.any():
            cr = (
                g.loc[mask_clean_selected, "num_dropped_relevant_clean"]
                / g.loc[mask_clean_selected, "num_selected_relevant_clean"]
            ).mean()
        else:
            cr = 0.0

        poison_drop_rates.append(pr * 100.0)
        clean_drop_rates.append(cr * 100.0)

    x = range(len(attack_types))
    width = 0.35

    plt.figure()
    plt.bar([i - width / 2 for i in x], poison_drop_rates, width=width, label="Poisoned docs dropped")
    plt.bar([i + width / 2 for i in x], clean_drop_rates, width=width, label="Clean docs dropped")

    plt.xticks(x, attack_types, rotation=20, ha="right")
    plt.ylabel("Average drop rate of relevant docs (%)")
    plt.title("Consensus Drop Behavior: Poison vs Clean (by attack)")
    plt.ylim(0, max(poison_drop_rates + clean_drop_rates + [10]) * 1.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "consensus_drop_behavior.png"))
    plt.close()


def plot_poison_volume_pipeline(all_df: pd.DataFrame, outdir: str):
    """
    For each attack_type, show % of relevant poisoned docs that:
      - are retrieved/selected by default
      - are selected by consensus
      - remain after consensus drops (kept)
    """
    grouped = all_df.groupby("attack_type")
    attack_types = list(grouped.groups.keys())

    default_retrieved_pct = []
    cons_selected_pct = []
    cons_kept_pct = []

    for atk, g in grouped:
        total_relevant_poison = g["num_relevant_poisoned"].sum()
        if total_relevant_poison == 0:
            default_retrieved_pct.append(0.0)
            cons_selected_pct.append(0.0)
            cons_kept_pct.append(0.0)
            continue

        default_retrieved = g["default_num_selected_relevant_poisoned"].sum()
        cons_selected = g["num_selected_relevant_poisoned"].sum()
        cons_kept = cons_selected - g["num_dropped_relevant_poisoned"].sum()

        default_retrieved_pct.append(default_retrieved / total_relevant_poison * 100.0)
        cons_selected_pct.append(cons_selected / total_relevant_poison * 100.0)
        cons_kept_pct.append(cons_kept / total_relevant_poison * 100.0)

    x = range(len(attack_types))
    width = 0.22

    plt.figure()
    plt.bar([i - width for i in x], default_retrieved_pct, width=width, label="Default: retrieved")
    plt.bar(x, cons_selected_pct, width=width, label="Consensus: selected")
    plt.bar([i + width for i in x], cons_kept_pct, width=width, label="Consensus: kept")

    plt.xticks(x, attack_types, rotation=20, ha="right")
    plt.ylabel("% of relevant poisoned docs")
    plt.title("Poison Volume Through Pipeline (doc counts normalized)")
    plt.ylim(0, max(default_retrieved_pct + cons_selected_pct + cons_kept_pct + [10]) * 1.2)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "poison_volume_pipeline.png"))
    plt.close()


def plot_relevant_drop_volume_by_attack(all_df: pd.DataFrame, outdir: str):
    """
    For each attack_type, show how many relevant docs were dropped in total,
    and how many of those were clean (unpoisoned).
    This lets you see which attacks cause the biggest absolute drop in evidence.
    (Consensus pipeline)
    """
    grouped = all_df.groupby("attack_type")
    attack_types = list(grouped.groups.keys())

    total_dropped_relevant = []
    total_dropped_clean = []

    for atk, g in grouped:
        total_dropped_relevant.append(g["num_dropped_relevant"].sum())
        total_dropped_clean.append(g["num_dropped_relevant_clean"].sum())

    x = range(len(attack_types))
    width = 0.35

    plt.figure()
    plt.bar([i - width / 2 for i in x], total_dropped_relevant, width=width, label="Total dropped relevant docs")
    plt.bar([i + width / 2 for i in x], total_dropped_clean, width=width, label="Dropped clean relevant docs")

    plt.xticks(x, attack_types, rotation=20, ha="right")
    plt.ylabel("Count of dropped relevant docs (summed across queries)")
    plt.title("Relevant Docs Dropped by Attack (Consensus pipeline)")
    ymax = max(total_dropped_relevant + total_dropped_clean + [1])
    plt.ylim(0, ymax * 1.2)
    for i, v in enumerate(total_dropped_relevant):
        plt.text(i - width / 2, v + 0.1, str(v), ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(total_dropped_clean):
        plt.text(i + width / 2, v + 0.1, str(v), ha="center", va="bottom", fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "relevant_drop_volume_by_attack.png"))
    plt.close()


def plot_relevant_recall_vs_ground_truth(all_df: pd.DataFrame, queries: dict, outdir: str):
    """
    For each attack_type, compute average recall of ground-truth relevant docs:
      - Default:   default_num_selected_relevant / |true_relevant|
      - Consensus: (num_selected_relevant - num_dropped_relevant) / |true_relevant|
    where |true_relevant| is len(doc_ids) from queries.jsonl for that query_id.
    """
    grouped = all_df.groupby("attack_type")
    attack_types = list(grouped.groups.keys())

    default_recalls = []
    consensus_recalls = []

    for atk, g in grouped:
        per_query_default = []
        per_query_consensus = []

        for _, row in g.iterrows():
            qid = str(row["query_id"])
            qinfo = queries.get(qid)
            if not qinfo:
                continue
            true_relevant = len(qinfo["doc_ids"])
            if true_relevant == 0:
                continue

            # Default recall
            if "default_num_selected_relevant" in row:
                default_cov = row["default_num_selected_relevant"]
                default_rec = min(default_cov / true_relevant, 1.0)
                per_query_default.append(default_rec)

            # Consensus recall (relevant docs kept)
            cons_selected_rel = row["num_selected_relevant"]
            cons_dropped_rel = row["num_dropped_relevant"]
            cons_kept_rel = cons_selected_rel - cons_dropped_rel
            cons_rec = min(cons_kept_rel / true_relevant, 1.0)
            per_query_consensus.append(cons_rec)

        if per_query_default:
            default_recalls.append(sum(per_query_default) / len(per_query_default) * 100.0)
        else:
            default_recalls.append(0.0)

        if per_query_consensus:
            consensus_recalls.append(sum(per_query_consensus) / len(per_query_consensus) * 100.0)
        else:
            consensus_recalls.append(0.0)

    x = range(len(attack_types))
    width = 0.35

    plt.figure()
    plt.bar([i - width / 2 for i in x], default_recalls, width=width, label="Default RAG")
    plt.bar([i + width / 2 for i in x], consensus_recalls, width=width, label="Consensus RAG")

    plt.xticks(x, attack_types, rotation=20, ha="right")
    plt.ylabel("Avg recall of ground-truth relevant docs (%)")
    plt.title("Relevant Coverage vs Ground Truth (by attack)")
    ymax = max(default_recalls + consensus_recalls + [1])
    plt.ylim(0, ymax * 1.2)
    for i, v in enumerate(default_recalls):
        plt.text(i - width / 2, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(consensus_recalls):
        plt.text(i + width / 2, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "relevant_recall_vs_ground_truth.png"))
    plt.close()


# ---------- Per-attack Charts ----------

def plot_per_attack_funnels(all_df: pd.DataFrame, outdir: str):
    """
    For each attack_type, create a small funnel-like bar chart comparing:
      Default:   P(retrieved), P(output poisoned)
      Consensus: P(retrieved), P(kept), P(output poisoned)
    """
    grouped = all_df.groupby("attack_type")

    for atk, g in grouped:
        # Default
        p_retrieved_def = (g["default_num_selected_relevant_poisoned"] > 0).mean() * 100.0
        p_output_def = g["default_agent_output_was_poisoned"].mean() * 100.0

        # Consensus
        p_retrieved_cons = (g["num_selected_relevant_poisoned"] > 0).mean() * 100.0
        p_kept_cons = ((g["num_selected_relevant_poisoned"] - g["num_dropped_relevant_poisoned"]) > 0).mean() * 100.0
        p_output_cons = g["consensus_agent_output_was_poisoned"].mean() * 100.0

        stages_def = ["retrieved", "output_poisoned"]
        vals_def = [p_retrieved_def, p_output_def]

        stages_cons = ["retrieved", "kept", "output_poisoned"]
        vals_cons = [p_retrieved_cons, p_kept_cons, p_output_cons]

        plt.figure()
        x_def = range(len(stages_def))
        x_cons = range(len(stages_cons))

        plt.subplot(1, 2, 1)
        plt.bar(x_def, vals_def)
        plt.xticks(x_def, stages_def, rotation=20, ha="right")
        plt.ylim(0, 110)
        plt.title("Default RAG")
        for i, v in enumerate(vals_def):
            plt.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=7)

        plt.subplot(1, 2, 2)
        plt.bar(x_cons, vals_cons)
        plt.xticks(x_cons, stages_cons, rotation=20, ha="right")
        plt.ylim(0, 110)
        plt.title("Consensus RAG")
        for i, v in enumerate(vals_cons):
            plt.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=7)

        plt.suptitle(f"Pipeline Funnel – Attack: {atk}")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fname = f"funnel_{atk}.png".replace(" ", "_")
        plt.savefig(os.path.join(outdir, fname))
        plt.close()


def plot_per_attack_drop_composition(all_df: pd.DataFrame, outdir: str):
    """
    For each attack_type, show how many dropped relevant docs were poisoned vs clean.
    (Consensus only)
    """
    grouped = all_df.groupby("attack_type")

    for atk, g in grouped:
        total_dropped_poison = g["num_dropped_relevant_poisoned"].sum()
        total_dropped_clean = g["num_dropped_relevant_clean"].sum()

        labels = ["Dropped clean", "Dropped poisoned"]
        values = [total_dropped_clean, total_dropped_poison]

        plt.figure()
        plt.bar(labels, values, color=["#4CAF50", "#F44336"])
        plt.ylabel("Count of dropped relevant docs")
        plt.title(f"Drop Composition – Attack: {atk}")
        for i, v in enumerate(values):
            plt.text(i, v + 0.1, str(v), ha="center", va="bottom")
        plt.tight_layout()
        fname = f"drop_composition_{atk}.png".replace(" ", "_")
        plt.savefig(os.path.join(outdir, fname))
        plt.close()


def plot_per_attack_outcome_grid(all_df: pd.DataFrame, outdir: str):
    """
    For each attack_type, categorize queries into:
      - both_clean
      - default_poisoned_consensus_clean
      - default_clean_consensus_poisoned
      - both_poisoned
    and plot as bar chart.
    """
    grouped = all_df.groupby("attack_type")

    for atk, g in grouped:
        categories = {
            "both_clean": 0,
            "default_poisoned_consensus_clean": 0,
            "default_clean_consensus_poisoned": 0,
            "both_poisoned": 0,
        }

        for _, row in g.iterrows():
            d_poison = row["default_agent_output_was_poisoned"]
            c_poison = row["consensus_agent_output_was_poisoned"]

            if not d_poison and not c_poison:
                categories["both_clean"] += 1
            elif d_poison and not c_poison:
                categories["default_poisoned_consensus_clean"] += 1
            elif not d_poison and c_poison:
                categories["default_clean_consensus_poisoned"] += 1
            else:  # both poisoned
                categories["both_poisoned"] += 1

        labels = list(categories.keys())
        values = [categories[k] for k in labels]

        plt.figure()
        plt.bar(labels, values)
        plt.ylabel("Number of queries")
        plt.title(f"Outcome Grid – Attack: {atk}")
        plt.xticks(rotation=20, ha="right")
        for i, v in enumerate(values):
            plt.text(i, v + 0.1, str(v), ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        fname = f"outcome_grid_{atk}.png".replace(" ", "_")
        plt.savefig(os.path.join(outdir, fname))
        plt.close()


def main():
    ensure_output_dir(OUTPUT_DIR)
    all_df = load_and_tag(CSV_FILES)
    queries = load_queries_jsonl(QUERIES_JSONL_PATH)

    # Combined charts
    plot_attack_success_comparison(all_df, OUTPUT_DIR)
    plot_poison_stage_survival(all_df, OUTPUT_DIR)
    plot_consensus_drop_behavior(all_df, OUTPUT_DIR)
    plot_poison_volume_pipeline(all_df, OUTPUT_DIR)
    plot_relevant_drop_volume_by_attack(all_df, OUTPUT_DIR)
    plot_relevant_recall_vs_ground_truth(all_df, queries, OUTPUT_DIR)

    # Per-attack charts
    plot_per_attack_funnels(all_df, OUTPUT_DIR)
    plot_per_attack_drop_composition(all_df, OUTPUT_DIR)
    plot_per_attack_outcome_grid(all_df, OUTPUT_DIR)

    print(f"Plots saved to folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()