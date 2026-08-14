# Forming embeddings of the jsons that we have created in the previous step. We will use the bge-m3 model to create embeddings of the text in the jsons.

import requests
import os
import json
import pandas as pd
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json = {
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()["embeddings"]
    return embedding

jsons = os.listdir("jsons/")
my_dicts = []
chunk_id = 0
for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating embeddings for {json_file}...")
    embeddings = create_embedding([c['text'] for c in content['chunks']])
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)

# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)
# print(df.head())

incoming_query = input("Ask your question: ")
question_embedding = create_embedding([incoming_query])[0]
# print(f"Embedding for the question: {question_embedding}")

similarities = cosine_similarity(np.vstack(df['embedding'].values), [question_embedding]).flatten()
top_results = 3
max_indx = similarities.argsort()[::-1][0:top_results]
print(f"Similarities: {max_indx}")
new_df = df.loc[max_indx]
print(new_df[['name', 'text', 'number']])

