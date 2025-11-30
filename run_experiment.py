# run_experiment implementation to execute one experiment 
# (poisoning, testing, statistic generation) with given hyperparameters

import pandas as pd
import json
import os
from dotenv import load_dotenv
from utils.openai_utils import embed_texts
from tqdm.auto import tqdm
import numpy as np

# local imports
from consensus_rag import consensus_rag_agent
from default_rag import default_rag_agent
from mutation import mutate_documents_with_malicious_prompt
from evaluation import generate_statistics, attack_success_judge

load_dotenv()

TOP_K_DOCS = 8
MAX_DOCS_TO_POISON_PER_QUERY = 1


# load in JSONL file and return as DataFrame
def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():  # skip blank lines
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

# converts docs to embedded dataframe
def embed_docs_df(docs):
    # extract / embed contents and keep only needed fields
    docs.loc[:, "embedding"] = embed_texts(docs.loc[:, "contents"].tolist())
    docs = docs.loc[:, ["doc_id", "contents", "poisoned", "embedding", "relevant_query_ids"]].copy()

    return docs


def run_experiment(
    attack_type="ignore_instructions",
    seed=123,
    top_k_docs=TOP_K_DOCS,
    max_docs_to_poison_per_query=MAX_DOCS_TO_POISON_PER_QUERY,
):
    # load documents + test questions for rag
    docs_df = load_jsonl(os.environ['DOCS_OUT_PATH'])
    questions_df = load_jsonl(os.environ['QUERIES_OUT_PATH'])

    # mutated docs contains new ['poisoned'] flag
    mutated_docs = mutate_documents_with_malicious_prompt(
        docs_df,
        questions_df,
        attack_type=attack_type,
        seed=seed,
        max_docs_to_poison_per_query=max_docs_to_poison_per_query,
    )

    embedded_docs = embed_docs_df(mutated_docs)

    all_stats = []

    # for each question run covid_question_agent and default_rag_agent
    for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Running RAG on default and consensus"):
        question = row.loc["query_text"]
        ground_truth = row.loc["ground_truth"]

        # default RAG
        print("DEFAULT RAG")
        default_rag_selected_docs, default_rag_response = default_rag_agent(embedded_docs, question, top_k_docs=top_k_docs)
        print(f"Q: {question}\nA: {default_rag_response}\n Selected Docs:\n{default_rag_selected_docs}\n\n")

        # consensus RAG
        print("CONSENSUS RAG")
        all_selected_docs, dropped_docs, consensus_rag_response = consensus_rag_agent(embedded_docs, question, top_k_docs=top_k_docs)
        print(f"Q: {question}\nA: {consensus_rag_response}\n Selected Docs:\n{all_selected_docs}\n Dropped Docs: {dropped_docs}\n\n")

        # attack success: default rag
        default_agent_used_poisoned_doc, default_agent_output_was_poisoned, default_answer_is_correct = attack_success_judge(ground_truth, default_rag_response, mutated_docs, default_rag_selected_docs)
        print(f"Used Poisoned Doc: {default_agent_used_poisoned_doc}\n Poisoned Output: {default_agent_output_was_poisoned}\n")

        # attack success: consensus rag
        consensus_agent_used_poisoned_doc, consensus_agent_output_was_poisoned, consensus_answer_is_correct = attack_success_judge(ground_truth, consensus_rag_response, mutated_docs, all_selected_docs, dropped_docs)
        print(f"Used Poisoned Doc: {consensus_agent_used_poisoned_doc}\n Poisoned Output: {consensus_agent_output_was_poisoned}\n")

        # per-query stats
        stats = generate_statistics(
            embedded_docs=embedded_docs,
            all_selected_docs=all_selected_docs,
            dropped_docs=dropped_docs,
            query_id=row["query_id"],
            default_rag_selected_docs=default_rag_selected_docs,
            default_agent_used_poisoned_doc=default_agent_used_poisoned_doc,
            default_agent_output_was_poisoned=default_agent_output_was_poisoned,
            consensus_agent_used_poisoned_doc=consensus_agent_used_poisoned_doc,
            consensus_agent_output_was_poisoned=consensus_agent_output_was_poisoned,
            default_answer_is_correct=default_answer_is_correct,
            consensus_answer_is_correct=consensus_answer_is_correct,
        )

        # attach metadata + raw outputs logging
        stats.update(
            {
                "attack_type": attack_type,
                "seed": seed,
                "top_k_docs": top_k_docs,
                "max_docs_to_poison_per_query": max_docs_to_poison_per_query,
                "question_text": question.replace("\n", ""),
                "ground_truth": ground_truth.replace("\n", ""),
                "default_rag_response": default_rag_response.replace("\n", ""),
                "consensus_rag_response": consensus_rag_response.replace("\n", ""),
            }
        )

        all_stats.append(stats)

    # save results
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv("experiment_stats.csv", index=False)
    print(stats_df.head())

    return stats_df


# if __name__ == "__main__":
#     run_experiment()
