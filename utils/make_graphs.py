import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# CONFIG – EDIT THESE
# ==========================
RESULTS_ROOT = "results"  # root dir containing experiment subfolders/CSVs
QUERIES_JSONL_PATH = "data/queries.jsonl"   # path to your queries jsonl
OUTPUT_DIR = "rag_attack_plots"
# ==========================


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_all_stats(results_root: str) -> pd.DataFrame:
    """
    Recursively load all .csv files under `results_root` and concatenate.

    Assumes each CSV has at least per-query stats, and ideally:
      - attack_type
      - seed
      - top_k_docs
      - max_docs_to_poison_per_query
    """
    dfs = []
    for dirpath, _, filenames in os.walk(results_root):
        for fname in filenames:
            if not fname.lower().endswith(".csv"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                df = pd.read_csv(fpath)

                # If attack_type isn't present, try to infer from folder name or filename
                if "attack_type" not in df.columns:
                    atk = None
                    segments = fpath.split(os.sep)
                    for seg in segments:
                        if seg.startswith("attack="):
                            atk = seg.split("=", 1)[1]
                            break
                    if atk is None:
                        atk = os.path.splitext(os.path.basename(fpath))[0]
                    df["attack_type"] = atk

                # Ensure query_id is string
                if "query_id" in df.columns:
                    df["query_id"] = df["query_id"].astype(str)

                # Cast booleans, if present
                bool_cols = [
                    "consensus_agent_used_poisoned_doc",
                    "consensus_agent_output_was_poisoned",
                    "default_agent_used_poisoned_doc",
                    "default_agent_output_was_poisoned",
                    "consensus_answer_is_correct",
                    "default_answer_is_correct",
                ]
                for col in bool_cols:
                    if col in df.columns:
                        # robust bool casting: treat 'true'/'1' as True
                        df[col] = df[col].map(lambda x: str(x).lower() in ("true", "1"))

                dfs.append(df)
            except Exception as e:
                print(f"Warning: failed to load {fpath}: {e}")

    if not dfs:
        raise RuntimeError(f"No CSV files found under {results_root}")
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


def plot_overall_accuracy_comparison(all_df: pd.DataFrame, outdir: str):
    """
    For each attack_type, compare:
      - % of queries where default answer matched ground truth (utility)
      - % of queries where consensus answer matched ground truth
    """
    if "default_answer_is_correct" not in all_df.columns or "consensus_answer_is_correct" not in all_df.columns:
        print("Skipping accuracy comparison: correctness columns not found.")
        return

    grouped = all_df.groupby("attack_type")
    attack_types = list(grouped.groups.keys())

    default_acc = []
    consensus_acc = []
    for atk, g in grouped:
        default_acc.append(g["default_answer_is_correct"].mean() * 100.0)
        consensus_acc.append(g["consensus_answer_is_correct"].mean() * 100.0)

    x = range(len(attack_types))
    width = 0.35

    plt.figure()
    plt.bar([i - width / 2 for i in x], default_acc, width=width, label="Default RAG")
    plt.bar([i + width / 2 for i in x], consensus_acc, width=width, label="Consensus RAG")

    plt.xticks(x, attack_types, rotation=20, ha="right")
    plt.ylabel("Answer correctness vs ground truth (%)")
    plt.title("Overall Answer Accuracy: Default vs Consensus")
    ymax = max(default_acc + consensus_acc + [1])
    plt.ylim(0, ymax * 1.2)
    for i, v in enumerate(default_acc):
        plt.text(i - width / 2, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(consensus_acc):
        plt.text(i + width / 2, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "answer_accuracy_default_vs_consensus.png"))
    plt.close()


# ---------- Hyperparameter sensitivity (combined across attacks) ----------

def _plot_metric_vs_hparam_combined(
    all_df: pd.DataFrame,
    metric_col_default: str,
    metric_col_consensus: str,
    hparam_col: str,
    ylabel: str,
    title: str,
    filename: str,
    outdir: str,
):
    """
    One combined plot (all attacks) for a metric vs hyperparameter:

    - x-axis: hyperparameter values
    - color: attack_type
    - linestyle: solid = Default, dashed = Consensus
    """
    if hparam_col not in all_df.columns:
        print(f"Skipping {title} plots: {hparam_col} not found in stats.")
        return

    attack_types = sorted(all_df["attack_type"].unique())
    if not attack_types:
        return

    # assign a color per attack
    cmap = plt.cm.get_cmap("tab10", len(attack_types))
    color_map = {atk: cmap(i) for i, atk in enumerate(attack_types)}

    plt.figure()

    for atk in attack_types:
        g = all_df[all_df["attack_type"] == atk]
        g = g.dropna(subset=[hparam_col])
        if g.empty:
            continue

        grouped = g.groupby(hparam_col)
        xs = sorted(grouped.groups.keys())

        default_vals = []
        cons_vals = []
        for x in xs:
            sub = grouped.get_group(x)
            default_vals.append(sub[metric_col_default].mean() * 100.0)
            cons_vals.append(sub[metric_col_consensus].mean() * 100.0)

        color = color_map[atk]
        plt.plot(xs, default_vals, marker="o", linestyle="-", color=color,
                 label=f"{atk} – Default")
        plt.plot(xs, cons_vals, marker="o", linestyle="--", color=color,
                 label=f"{atk} – Consensus")

    plt.xlabel(hparam_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # avoid legend duplicates if any
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    plt.legend(uniq.values(), uniq.keys(), fontsize=8)

    # scale y
    all_y = []
    for line in plt.gca().get_lines():
        all_y.extend(line.get_ydata())
    if all_y:
        ymax = max(all_y + [1])
        plt.ylim(0, ymax * 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename))
    plt.close()


def plot_hparam_sensitivity(all_df: pd.DataFrame, outdir: str):
    """
    Create combined sensitivity plots (all attacks together):

      - ASR vs docs poisoned (max_docs_to_poison_per_query)
      - Accuracy vs docs poisoned
      - ASR vs top-k
      - Accuracy vs top-k
    """
    # ASR vs # poisoned docs
    _plot_metric_vs_hparam_combined(
        all_df=all_df,
        metric_col_default="default_agent_output_was_poisoned",
        metric_col_consensus="consensus_agent_output_was_poisoned",
        hparam_col="max_docs_to_poison_per_query",
        ylabel="Poisoned output rate (%)",
        title="ASR vs docs poisoned",
        filename="asr_vs_docs_poisoned.png",
        outdir=outdir,
    )

    # Accuracy vs # poisoned docs
    if "default_answer_is_correct" in all_df.columns and "consensus_answer_is_correct" in all_df.columns:
        _plot_metric_vs_hparam_combined(
            all_df=all_df,
            metric_col_default="default_answer_is_correct",
            metric_col_consensus="consensus_answer_is_correct",
            hparam_col="max_docs_to_poison_per_query",
            ylabel="Answer correctness vs ground truth (%)",
            title="Accuracy vs docs poisoned",
            filename="accuracy_vs_docs_poisoned.png",
            outdir=outdir,
        )

    # ASR vs top_k_docs
    _plot_metric_vs_hparam_combined(
        all_df=all_df,
        metric_col_default="default_agent_output_was_poisoned",
        metric_col_consensus="consensus_agent_output_was_poisoned",
        hparam_col="top_k_docs",
        ylabel="Poisoned output rate (%)",
        title="ASR vs top-k",
        filename="asr_vs_topk.png",
        outdir=outdir,
    )

    # Accuracy vs top_k_docs
    if "default_answer_is_correct" in all_df.columns and "consensus_answer_is_correct" in all_df.columns:
        _plot_metric_vs_hparam_combined(
            all_df=all_df,
            metric_col_default="default_answer_is_correct",
            metric_col_consensus="consensus_answer_is_correct",
            hparam_col="top_k_docs",
            ylabel="Answer correctness vs ground truth (%)",
            title="Accuracy vs top-k",
            filename="accuracy_vs_topk.png",
            outdir=outdir,
        )


# ---------- Document-level behavior (aggregated over hyperparams) ----------

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


# ---------- Per-attack Funnels (combined into one figure) ----------

def plot_per_attack_funnels(all_df: pd.DataFrame, outdir: str):
    """
    One figure with rows = attack types, columns = [Default, Consensus].

    For each attack_type, create a funnel-like bar chart comparing:
      Default:   P(retrieved), P(output poisoned)
      Consensus: P(retrieved), P(kept), P(output poisoned)
    """
    grouped = all_df.groupby("attack_type")
    attack_types = sorted(grouped.groups.keys())
    if not attack_types:
        return

    n_attacks = len(attack_types)
    fig, axes = plt.subplots(
        nrows=n_attacks,
        ncols=2,
        figsize=(10, 3 * n_attacks),
        squeeze=False,
    )

    for row_idx, atk in enumerate(attack_types):
        g = grouped.get_group(atk)

        # Default
        p_retrieved_def = (g["default_num_selected_relevant_poisoned"] > 0).mean() * 100.0
        p_output_def = g["default_agent_output_was_poisoned"].mean() * 100.0

        stages_def = ["retrieved", "output_poisoned"]
        vals_def = [p_retrieved_def, p_output_def]

        ax_def = axes[row_idx, 0]
        x_def = range(len(stages_def))
        ax_def.bar(x_def, vals_def)
        ax_def.set_xticks(list(x_def))
        ax_def.set_xticklabels(stages_def, rotation=20, ha="right")
        ax_def.set_ylim(0, 110)
        ax_def.set_title(f"{atk} – Default RAG")
        for i, v in enumerate(vals_def):
            ax_def.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=7)

        # Consensus
        p_retrieved_cons = (g["num_selected_relevant_poisoned"] > 0).mean() * 100.0
        p_kept_cons = ((g["num_selected_relevant_poisoned"] - g["num_dropped_relevant_poisoned"]) > 0).mean() * 100.0
        p_output_cons = g["consensus_agent_output_was_poisoned"].mean() * 100.0

        stages_cons = ["retrieved", "kept", "output_poisoned"]
        vals_cons = [p_retrieved_cons, p_kept_cons, p_output_cons]

        ax_cons = axes[row_idx, 1]
        x_cons = range(len(stages_cons))
        ax_cons.bar(x_cons, vals_cons)
        ax_cons.set_xticks(list(x_cons))
        ax_cons.set_xticklabels(stages_cons, rotation=20, ha="right")
        ax_cons.set_ylim(0, 110)
        ax_cons.set_title(f"{atk} – Consensus RAG")
        for i, v in enumerate(vals_cons):
            ax_cons.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Pipeline Funnels by Attack", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = "funnels_by_attack.png"
    fig.savefig(os.path.join(outdir, fname))
    plt.close(fig)


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

def plot_poisoned_docs_vs_topk_and_budget(all_df: pd.DataFrame, outdir: str):
    """
    Attack Success Rate (ASR) vs top-k and docs poisoned.

    For each combination of:
        - top_k_docs
        - max_docs_to_poison_per_query
    compute:

        ASR_default   = mean(default_agent_output_was_poisoned) * 100
        ASR_consensus = mean(consensus_agent_output_was_poisoned) * 100

    and show them as two heatmaps:

        [ Default RAG ASR | Consensus RAG ASR | Colorbar ]
    """

    required_cols = [
        "top_k_docs",
        "max_docs_to_poison_per_query",
        "default_agent_output_was_poisoned",
        "consensus_agent_output_was_poisoned",
    ]
    for c in required_cols:
        if c not in all_df.columns:
            print(f"Skipping ASR heatmap: missing column '{c}'.")
            return

    # ---- group by hyperparameters and compute ASR ----
    grouped_def = (
        all_df
        .groupby(["max_docs_to_poison_per_query", "top_k_docs"])["default_agent_output_was_poisoned"]
        .mean()
        .reset_index()
    )
    grouped_cons = (
        all_df
        .groupby(["max_docs_to_poison_per_query", "top_k_docs"])["consensus_agent_output_was_poisoned"]
        .mean()
        .reset_index()
    )

    # pivot to 2D grids, convert to %
    pivot_default = (
        grouped_def
        .pivot(index="max_docs_to_poison_per_query", columns="top_k_docs",
               values="default_agent_output_was_poisoned")
        * 100.0
    )
    pivot_cons = (
        grouped_cons
        .pivot(index="max_docs_to_poison_per_query", columns="top_k_docs",
               values="consensus_agent_output_was_poisoned")
        * 100.0
    )

    pivot_default = pivot_default.sort_index(axis=0).sort_index(axis=1).fillna(0.0)
    pivot_cons = pivot_cons.sort_index(axis=0).sort_index(axis=1).fillna(0.0)

    if pivot_default.empty or pivot_cons.empty:
        print("Skipping ASR heatmap: empty pivot.")
        return

    # shared color scale across both heatmaps
    vmax = max(pivot_default.values.max(), pivot_cons.values.max())
    vmax = max(vmax, 1.0)  # avoid degenerate scale

    # ---- FIGURE & AXES: 1 row, 3 columns (last is colorbar) ----
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(9, 3),
        gridspec_kw={"width_ratios": [1, 1, 0.05]},
    )
    ax_def, ax_cons, cbar_ax = axes
    cmap = "viridis"

    # ---------- Default RAG ASR heatmap ----------
    im_def = ax_def.imshow(
        pivot_default.values,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
    )
    ax_def.set_title("Default RAG ASR", fontsize=11)
    ax_def.set_xlabel("Top-k")
    ax_def.set_xticks(range(len(pivot_default.columns)))
    ax_def.set_xticklabels(pivot_default.columns)
    ax_def.set_ylabel("# Docs Poisoned")
    ax_def.set_yticks(range(len(pivot_default.index)))
    ax_def.set_yticklabels(pivot_default.index)

    mean_def = pivot_default.values.mean()
    for i, yval in enumerate(pivot_default.index):
        for j, xval in enumerate(pivot_default.columns):
            val = pivot_default.iloc[i, j]
            text_color = "white" if val > mean_def else "black"
            ax_def.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7, color=text_color)

    # ---------- Consensus RAG ASR heatmap ----------
    im_cons = ax_cons.imshow(
        pivot_cons.values,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
    )
    ax_cons.set_title("Consensus RAG ASR", fontsize=11)
    ax_cons.set_xlabel("Top-k")
    ax_cons.set_xticks(range(len(pivot_cons.columns)))
    ax_cons.set_xticklabels(pivot_cons.columns)
    ax_cons.set_ylabel("")  # share y with left heatmap
    ax_cons.set_yticks(range(len(pivot_cons.index)))
    ax_cons.set_yticklabels(pivot_cons.index)

    mean_cons = pivot_cons.values.mean()
    for i, yval in enumerate(pivot_cons.index):
        for j, xval in enumerate(pivot_cons.columns):
            val = pivot_cons.iloc[i, j]
            text_color = "white" if val > mean_cons else "black"
            ax_cons.text(j, i, f"{val:.1f}", ha="center", va="center",
                         fontsize=7, color=text_color)

    # ---------- Colorbar in its own narrow axis ----------
    cbar = fig.colorbar(im_cons, cax=cbar_ax)
    cbar.set_label("Attack success rate (%)", fontsize=9)

    fig.suptitle("Attack success vs Top-k and Docs Poisoned", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = os.path.join(outdir, "asr_vs_topk_and_docs_poisoned.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ensure_output_dir(OUTPUT_DIR)
    all_df = load_all_stats(RESULTS_ROOT)
    queries = load_queries_jsonl(QUERIES_JSONL_PATH)

    # Combined charts (global summary)
    plot_attack_success_comparison(all_df, OUTPUT_DIR)
    plot_overall_accuracy_comparison(all_df, OUTPUT_DIR)

    # Hyperparameter sensitivity charts (combined)
    plot_hparam_sensitivity(all_df, OUTPUT_DIR)

    # New 2D interaction plot: top-k vs poisoning budget
    plot_poisoned_docs_vs_topk_and_budget(all_df, OUTPUT_DIR)

    # Document-level behavior
    plot_consensus_drop_behavior(all_df, OUTPUT_DIR)
    plot_poison_volume_pipeline(all_df, OUTPUT_DIR)
    plot_relevant_drop_volume_by_attack(all_df, OUTPUT_DIR)
    plot_relevant_recall_vs_ground_truth(all_df, queries, OUTPUT_DIR)

    # Per-attack charts
    plot_per_attack_funnels(all_df, OUTPUT_DIR)
    plot_per_attack_outcome_grid(all_df, OUTPUT_DIR)

    print(f"Plots saved to folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()