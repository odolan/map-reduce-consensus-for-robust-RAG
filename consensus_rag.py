# consensus-map-reduce system for robust RAG -- new defense

import numpy as np

# local imports
from utils.openai_utils import query_documents, generate_llm_response, get_embedding, cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.rate_limiter import RateLimiter


# helper used threading pool to reassemble results in order.
def _process_single_doc(idx, question, doc):
    try:
        result = doc_processor_agent(question, doc)
    except Exception as e:
        result = f"ERROR processing document index {idx}: {e}"
    return idx, result


LLM_RATE_LIMITER = RateLimiter(max_calls_per_minute=500) # global rate limiter instance
MAX_DOC_WORKERS = 16 # how many docs to summarize in parallel at most

# sub agent - processes / summarizes a document as it relates to the question (thread safe)
def doc_processor_agent(question, doc):

    system_msg = (
        "You are a helpful assistant assigned to one document. Your task is to read the "
        "document and provide a concise answer to the user's question based solely on "
        "the information in the document. If the document does not contain relevant "
        "information, respond with 'The document does not contain relevant information "
        "to answer the question.'"
    )
    prompt = (
        f"Using the following document, answer the question concisely.\num_embeddings\num_embeddings"
        f"Document:\num_embeddings{doc}\num_embeddings\nQuestion: {question}\nAnswer:"
    )

    # respect global 500 RPM limit
    LLM_RATE_LIMITER.wait_for_slot()

    response = generate_llm_response(system_msg=system_msg, prompt=prompt)
    return response


# produces consensus response from multiple sub-agent outputs 
# takes a list of sub-agent responses, embeds them, finds their pairwise similarities, and drops outliers.
# Returns: indices into responses that were dropped, consensus_answer
def get_consensus_response(responses, model="text-embedding-3-small", std_mult=1.0):

    if not responses:
        return [], "No responses were provided."

    # embed sub agent responses
    embeddings = [get_embedding(r, model=model) for r in responses]
    num_embeddings = len(embeddings)

    # default case (only one doc pulled)
    if num_embeddings == 1:
        return [], responses[0]

    # compute pairwise cosine similarities
    similarities = np.zeros((num_embeddings, num_embeddings))

    # compute pairwise similarities
    for doc_a in range(num_embeddings):
        for doc_b in range(num_embeddings):
            similarities[doc_a, doc_b] = cosine_similarity(embeddings[doc_a], embeddings[doc_b])

    # average similarity per response (excluding self)
    avg_similarities = []
    for doc_a in range(num_embeddings):
        row_sum = similarities[doc_a].sum() - similarities[doc_a, doc_a]
        avg_similarities.append(row_sum / (num_embeddings - 1))

    avg_similarities = np.array(avg_similarities)

    # find the mean and std of similarities to threshold outliers
    mean_similarity = float(np.mean(avg_similarities))
    std_similarity = float(np.std(avg_similarities))

    # if no docs stand out font drop any
    if std_similarity == 0.0 or num_embeddings < 4:
        kept_indices = list(range(num_embeddings))
        dropped_indices = []
    else:
        threshold = mean_similarity - std_mult * std_similarity

        # documents that stand out from the rest past some threshold
        candidate_outliers = [doc_a for doc_a, s in enumerate(avg_similarities) if s < threshold]

        # drop clear outliers
        if len(candidate_outliers) > 0 and len(candidate_outliers) < num_embeddings - 1:
            dropped_indices = candidate_outliers
            kept_indices = [doc_a for doc_a in range(num_embeddings) if doc_a not in dropped_indices]
        else:
            dropped_indices = []
            kept_indices = list(range(num_embeddings))

    kept_responses = [responses[doc_a] for doc_a in kept_indices]

    system_msg = (
        "You are an agent tasked with aggregating multiple answers from different agents sub-agents. "
        "Some may be noisy or contradictory. Produce a concise, factually consistent consensus answer based only on the information "
        "that is shared across most of the answers."
    )

    prompt = "Here are the answers from several agents:\num_embeddings\num_embeddings"
    for idx, r in enumerate(kept_responses):
        prompt += f"Answer {idx+1}:\num_embeddings{r}\num_embeddings\num_embeddings"
    prompt += "Please provide a single concise consensus answer that reflects the majority view."

    # rate-limit the consensus LLM call as well
    LLM_RATE_LIMITER.wait_for_slot()
    consensus_answer = generate_llm_response(system_msg=system_msg, prompt=prompt)

    return dropped_indices, consensus_answer


# reducer agent - collects results from the sub agents and votes on consensus
# returns used_doc_ids, dropped_docs, consensus_response
def consensus_rag_agent(docs, question, top_k_docs=5):

    # query documents to get top k most similar
    relevant_docs = query_documents(docs, question, top_k=top_k_docs)

    # extract the doc contents
    relevant_docs_content = relevant_docs.loc[:, "contents"].tolist()

    # assign sub-agents in parallel to process each document that needs to be summarized
    num_docs = len(relevant_docs_content)
    responses = [None] * num_docs

    with ThreadPoolExecutor(max_workers=MAX_DOC_WORKERS) as executor:
        # launch one task per document
        futures = [executor.submit(_process_single_doc, idx, question, doc) for idx, doc in enumerate(relevant_docs_content)]

        # collect results as they finish
        for future in as_completed(futures):
            idx, res = future.result()
            responses[idx] = res

    dropped_indices, consensus_response = get_consensus_response(responses)

    # map dropped indices back to doc_ids
    dropped_docs = relevant_docs.iloc[dropped_indices]["doc_id"].tolist()

    # return doc_ids used, documents dropped by consensus (poisoned or unrelated), consensus response
    return relevant_docs.loc[:, "doc_id"], dropped_docs, consensus_response
