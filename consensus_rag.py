# this is the map-reduce-consensus approach for robust RAG on COVID-19 questions

from utils.openai_utils import query_documents, generate_llm_response, get_embedding, cosine_similarity
import pandas as pd
import numpy as np


# sub agent - processes / summarizes a document
def doc_processor_agent(question, doc):
	# summarize output as it relates to question 
	system_msg = "You are a helpful assistant assigned to one document. Your task is to read the document and provide a concise answer to the user's question based solely on the information in the document. If the document does not contain relevant information, respond with 'The document does not contain relevant information to answer the question.'"
	prompt = f"Using the following document, answer the question concisely.\n\nDocument:\n{doc}\n\nQuestion: {question}\nAnswer:"
	response = generate_llm_response(system_msg=system_msg, prompt=prompt)
	return response


# produces consensus response from multiple sub-agent outputs
def get_consensus_response(responses, model="text-embedding-3-small", std_mult=1.0):
    """
    Given a list of sub-agent responses this function computes pairwise similarities 
    keeping responses that are most similar to others (majority cluster) while dropping outliers

    Returns:
      dropped_indices: list[int]  # indices into `responses` that were dropped
      consensus_answer: str
    """
    if not responses:
        return [], "No responses were provided."

    # embed sub agent responses
    embeddings = [get_embedding(r, model=model) for r in responses]
    n = len(embeddings)

    if n == 1:
        return [], responses[0]

    # compute pairwise cosine similarities
    sims = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sims[i, j] = cosine_similarity(embeddings[i], embeddings[j])

    # average similarity per response (excluding self)
    avg_sims = []
    for i in range(n):
        row_sum = sims[i].sum() - sims[i, i]
        avg_sims.append(row_sum / (n - 1))
    avg_sims = np.array(avg_sims)

    # ----- NEW: dynamic outlier detection -----
    mean_sim = float(np.mean(avg_sims))
    std_sim = float(np.std(avg_sims))

    # if everything looks basically the same, don't drop anything
    if std_sim == 0.0 or n < 4:
        kept_indices = list(range(n))
        dropped_indices = []
    else:
        threshold = mean_sim - std_mult * std_sim

        # candidates that are significantly less similar than the rest
        candidate_outliers = [i for i, s in enumerate(avg_sims) if s < threshold]

        # Only drop clearly "few" outliers; otherwise keep all
        if 1 <= len(candidate_outliers) <= 2:
            dropped_indices = candidate_outliers
            kept_indices = [i for i in range(n) if i not in dropped_indices]
        else:
            dropped_indices = []
            kept_indices = list(range(n))
    # ------------------------------------------

    kept_responses = [responses[i] for i in kept_indices]

    system_msg = (
        "You are aggregating multiple answers from different agents. "
        "Some may be noisy or contradictory. Produce a concise, "
        "factually consistent consensus answer based only on the information "
        "that is shared across most of the answers."
    )

    prompt = "Here are the answers from several agents:\n\n"
    for idx, r in enumerate(kept_responses):
        prompt += f"Answer {idx+1}:\n{r}\n\n"
    prompt += "Please provide a single concise consensus answer that reflects the majority view."

    consensus_answer = generate_llm_response(system_msg=system_msg, prompt=prompt)

    return dropped_indices, consensus_answer


# reducer agent - collects results from the sub agents and votes on consensus
def consensus_rag_agent(docs, question, top_k_docs=5):

	# query documents to get top k most similar
	relevant_docs = query_documents(docs, question, top_k=top_k_docs)

	# extract the doc contents
	relevant_docs_content = relevant_docs.loc[:, "contents"].tolist()
	responses = [doc_processor_agent(question, doc) for doc in relevant_docs_content]

	dropped_indices, consensus_response = get_consensus_response(responses)

	# map dropped indices back to doc_ids
	dropped_docs = relevant_docs.iloc[dropped_indices]["doc_id"].tolist()

	# return doc_ids used, documents dropped by consensus (poisoned or unrelated), consensus response
	return relevant_docs.loc[:,"doc_id"], dropped_docs, consensus_response
