# Forming embeddings of the jsons that we have created in the previous step. We will use the bge-m3 model to create embeddings of the text in the jsons.

# read_chunks.py
import json
import os
import joblib
import numpy as np
import pandas as pd
import requests


def create_embedding(text_list):
  r = requests.post(
      "http://localhost:11434/api/embed",
      json={"model": "bge-m3", "input": text_list},
  )
  return r.json()["embeddings"]


jsons = os.listdir("jsons/")
my_dicts = []
chunk_id = 0
for json_file in jsons:
  with open(f"jsons/{json_file}") as f:
    content = json.load(f)
  print(f"Creating embeddings for {json_file}...")
  embeddings = create_embedding([c["text"] for c in content["chunks"]])
  for i, chunk in enumerate(content["chunks"]):
    chunk["chunk_id"] = chunk_id
    chunk["embedding"] = embeddings[i]
    chunk_id += 1
    my_dicts.append(chunk)

df = pd.DataFrame.from_records(my_dicts)
joblib.dump(df, "embeddings.joblib")
print("Saved embeddings.joblib successfully!")