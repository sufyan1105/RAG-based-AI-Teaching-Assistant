import os

import joblib
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "embeddings.joblib")
TOP_K = int(os.getenv("TOP_K", "5"))


def create_embedding(text_list):
    r = requests.post(f"{OLLAMA_URL}/api/embed", json={"model": EMBED_MODEL, "input": text_list})
    r.raise_for_status()
    return r.json()["embeddings"]


def inference(prompt):
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
    )
    r.raise_for_status()
    return r.json()["response"]


def to_mmss(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def retrieve(df, query, k=TOP_K):
    q = create_embedding([query])[0]
    sims = cosine_similarity(np.vstack(df["embedding"].values), [q]).flatten()
    return df.iloc[sims.argsort()[::-1][:k]]


def answer(df, query):
    chunks = retrieve(df, query)
    records = chunks[["name", "text", "number", "start", "end"]].assign(
        start=chunks["start"].apply(to_mmss), end=chunks["end"].apply(to_mmss)
    ).to_json(orient="records")
    prompt = f''' I am teaching web development course. Here are video chunks containing video names, text, number, start_time , end_time (in mm:ss) an the text at that time .:
{records}
-----------------------------------------------------------
" {query} "
User asked this question related to the video chunks, you have to answer where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that part. If user asks unrelated questions tell him you can only answer questions related to the Course. Keep the answer short and precise: 2-4 lines, just the video name, number and timestamps.
'''
    return inference(prompt), chunks


if __name__ == "__main__":
    df = joblib.load(EMBEDDINGS_PATH)
    while True:
        try:
            query = input("\nAsk your question (empty to quit): ").strip()
        except EOFError:
            break
        if not query:
            break
        response, _ = answer(df, query)
        print(f"Response: {response}")
        with open("response.txt", "w") as f:
            f.write(response)
