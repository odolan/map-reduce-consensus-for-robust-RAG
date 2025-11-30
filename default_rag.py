# default-rag system implementation (naive aproach without consensus for baseline)

import numpy as np

# local imports
from utils.openai_utils import query_documents, generate_llm_response


# default RAG agent function queries documents, processes, and generates responses
def default_rag_agent(embeded_docs, question, top_k_docs=5):

    # query documents to get top k most similar
    relevant_docs = query_documents(embeded_docs, question, top_k_docs)
    relevant_docs_contents = relevant_docs.loc[:, "contents"].tolist()

    # construct prompt for LLM
    prompt = "Use the following documents to answer the question. You should select the most relevant piece of information and susinctly convey it to the user:\n\n"
    for idx, doc in enumerate(relevant_docs_contents):
        prompt += f"{doc}\n\n"
    prompt += f"Question: {question}\nAnswer:"

    # generate LLM response
    response = generate_llm_response(prompt)
    return relevant_docs.loc[:, "doc_id"], response