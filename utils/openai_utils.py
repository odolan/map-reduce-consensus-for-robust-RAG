from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import os

load_dotenv()

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


# embed list of texts using OpenAI embeddings
def embed_texts(text_list, model="text-embedding-3-small"):
    response = client.embeddings.create(
        input=text_list,
        model=model
    )
    return [item.embedding for item in response.data]


# get embedding for single text
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    resp = client.embeddings.create(
        input=[text],
        model=model
    )
    return resp.data[0].embedding

# compute cosine similarity between two vectors
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# query documents and return top k most similar
def query_documents(df, query, top_k=5):
    query_emb = get_embedding(query, model="text-embedding-3-small")

    df.loc[:,"similarities"] = df.loc[:,"embedding"].apply(lambda emb: cosine_similarity(emb, query_emb))
    return df.sort_values("similarities", ascending=False).head(top_k)


def generate_llm_response(prompt, system_msg=None):
    messages = []

    # add a system message if provided
    if system_msg:
        messages.append({
            "role": "system",
            "content": system_msg
        })

    messages.append({"role": "user", "content": prompt})

    # get llm response
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages
    )

    return resp.choices[0].message.content