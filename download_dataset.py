"""
Create a ~500-document TREC-COVID subset for map-reduce consensus RAG.
"""

import json
from collections import defaultdict

import ir_datasets

DOCS_OUT_PATH = "covid_docs.jsonl"
QUERIES_OUT_PATH = "covid_queries.jsonl"

MAX_QUERIES = 50   # at most 50 queries/questions
DOCS_PER_QUERY = 10  # top 10 docs per selected query


"""Load the BEIR TREC-COVID dataset using ir_datasets."""
def load_trec_covid_dataset():

    dataset_id = "beir/trec-covid"
    dataset = ir_datasets.load(dataset_id)
    print(f"Loaded dataset: {dataset_id}")
    return dataset


"""
Build a mapping: query_id -> list of (doc_id, relevance)
uses the dataset's qrels_iter() which yields objects
with query_id, doc_id, relevance, iteration
"""
def build_qrels_by_query_id(dataset):
    
    query_id_to_qrels = defaultdict(list)

    for qrel in dataset.qrels_iter():
        query_id_to_qrels[qrel.query_id].append((qrel.doc_id, qrel.relevance))

    return query_id_to_qrels


"""
Return sorted list of query_ids that have at least DOCS_PER_QUERY relevant documents
"""
def select_queries_with_enough_relevant_docs(query_id_to_qrels):
    
    qualified_query_ids = []

    for query_id, doc_list in query_id_to_qrels.items():

        # count how many docs have relevance > 0
        num_relevant_docs = sum(1 for (_, relevance) in doc_list if relevance > 0)

        if num_relevant_docs >= DOCS_PER_QUERY:
            qualified_query_ids.append(query_id)

    # sort and return ids
    qualified_query_ids = sorted(qualified_query_ids)
    return qualified_query_ids

""" For each query_id, choose the top DOCS_PER_QUERY relevant documents (sorted)"""
def choose_top_docs_per_query(query_ids, query_id_to_qrels):
    query_id_to_doc_ids = {}
    doc_id_to_query_ids = defaultdict(list)

    for query_id in query_ids:
        qrels_for_query = query_id_to_qrels[query_id]

        # filter only relevant docs
        relevant_doc_pairs = [
            (doc_id, relevance)
            for (doc_id, relevance) in qrels_for_query
            if relevance > 0
        ]

        # sort by relevance descending then doc_id
        relevant_doc_pairs.sort(key=lambda pair: (-pair[1], pair[0]))

        # keep top DOCS_PER_QUERY doc_ids (removing least relevant docs for each question)
        top_doc_ids = [doc_id for (doc_id, _) in relevant_doc_pairs[:DOCS_PER_QUERY]]
        query_id_to_doc_ids[query_id] = top_doc_ids

        # fill the reverse mapping doc_id -> query_ids
        for doc_id in top_doc_ids:
            doc_id_to_query_ids[doc_id].append(query_id)

    return query_id_to_doc_ids, doc_id_to_query_ids

""" Collect only selected documents from dataset.docs_iter() and make dictionary"""
def build_doc_lookup_for_selected_docs(dataset, selected_doc_ids):
    doc_id_to_doc = {}
    selected_doc_ids = set(selected_doc_ids)

    print("selecting documents from docs_iter...")

    for doc in dataset.docs_iter():
        if doc.doc_id in selected_doc_ids:
            doc_id_to_doc[doc.doc_id] = doc

            if len(doc_id_to_doc) == len(selected_doc_ids):
                break

    return doc_id_to_doc

# write selected docs to JSONL file
# each line has: doc_id, title, text, contents, relevant_query_ids
def write_docs_jsonl(doc_id_to_doc, doc_id_to_query_ids, output_path):
    print(f"Writing docs to: ", output_path)

    with open(output_path, "w", encoding="utf-8") as docs_file:
        for doc_id in sorted(doc_id_to_query_ids.keys()):

            doc = doc_id_to_doc.get(doc_id)

            # skip docs that wont load
            if doc is None:
                continue

            title = getattr(doc, "title", "") or ""
            text = getattr(doc, "text", "") or ""

            # combine title + text for to make contents
            contents = (title + "\n\n" + text).strip()

            record = {
                "doc_id": doc_id,
                "title": title,
                "text": text,
                "contents": contents,
                "relevant_query_ids": sorted(doc_id_to_query_ids[doc_id]),
            }

            docs_file.write(json.dumps(record) + "\n")


# build lookup table: to convert query_id to query_text
def build_query_text_lookup(dataset, query_ids):
    query_id_to_text = {}
    target_query_ids = set(query_ids)

    print("collecting query texts for selected queries...")

    for query in dataset.queries_iter():
        if query.query_id in target_query_ids:
            query_id_to_text[query.query_id] = query.text

            # exit when we have all of them
            if len(query_id_to_text) == len(target_query_ids):
                break

    return query_id_to_text


# write selected queries to JSONL
# each line has: query_id, query_text, doc_ids
def write_queries_jsonl(selected_query_ids, query_id_to_text, query_id_to_doc_ids, output_path):

    print(f"Writing queries to: ", output_path)

    with open(output_path, "w", encoding="utf-8") as queries_file:
        for query_id in sorted(selected_query_ids):
            query_text = query_id_to_text.get(query_id, "")
            doc_ids = query_id_to_doc_ids.get(query_id, [])

            record = {
                "query_id": query_id,
                "query_text": query_text,
                "doc_ids": doc_ids,
            }

            queries_file.write(json.dumps(record) + "\n")


def main():

    # 1: load the covid dataset
    dataset = load_trec_covid_dataset()

    # 2: create qrels mapping query_id -> list of (doc_id, relevance)
    query_id_to_qrels = build_qrels_by_query_id(dataset)

    # 3: select queries with enough relevant documents
    queries_with_enough_docs = select_queries_with_enough_relevant_docs(query_id_to_qrels)
    print(f"Found {len(queries_with_enough_docs)} queries with ≥ {DOCS_PER_QUERY} relevant docs.")

    # limit to at most MAX_QUERIES queries
    selected_query_ids = queries_with_enough_docs[:MAX_QUERIES]
    print(f"Selecting {len(selected_query_ids)} queries for the subset.")

    # 4: for each selected query, choose the top DOCS_PER_QUERY relevant docs
    query_id_to_doc_ids, doc_id_to_query_ids = choose_top_docs_per_query(
        selected_query_ids,
        query_id_to_qrels,
    )

    selected_doc_ids = set(doc_id_to_query_ids.keys())
    print(f"Total unique docs selected: {len(selected_doc_ids)} "
          f"(target ≈ {MAX_QUERIES * DOCS_PER_QUERY}).")

    # 5: Build a lookup table
    doc_id_to_doc = build_doc_lookup_for_selected_docs(dataset, selected_doc_ids)

    # 6: write documents JSONL
    write_docs_jsonl(doc_id_to_doc, doc_id_to_query_ids, DOCS_OUT_PATH)

    # 7: collect query text for queries
    query_id_to_text = build_query_text_lookup(dataset, selected_query_ids)

    # 8: Make queries JSONL
    write_queries_jsonl(selected_query_ids, query_id_to_text, query_id_to_doc_ids, QUERIES_OUT_PATH)

    print(f"Download Complete")


if __name__ == "__main__":
    main()