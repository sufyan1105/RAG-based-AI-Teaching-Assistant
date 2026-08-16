# process_incomming.py
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity


def create_embedding(text_list):
  r = requests.post(
      "http://localhost:11434/api/embed",
      json={"model": "bge-m3", "input": text_list},
  )
  return r.json()["embeddings"]


if __name__ == "__main__":
  df = joblib.load("embeddings.joblib")

  incoming_query = input("Ask your question: ")
  question_embedding = create_embedding([incoming_query])[0]

  similarities = cosine_similarity(
      np.vstack(df["embedding"].values), [question_embedding]
  ).flatten()
  top_results = 3
  max_indx = similarities.argsort()[::-1][0:top_results]
  print(f"Similarities: {max_indx}")
  new_df = df.loc[max_indx]
  print(new_df[["name", "text", "number"]])