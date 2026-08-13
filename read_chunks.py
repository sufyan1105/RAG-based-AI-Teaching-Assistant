# Forming embeddings of the jsons that we have created in the previous step. We will use the bge-m3 model to create embeddings of the text in the jsons.

import requests
import os
import json

def create_embedding(text):
    r = requests.post("http://localhost:11434/api/embeddings", json = {
        "model": "bge-m3",
        "prompt": text
    })
    embedding = r.json()["embedding"]
    return embedding

jsons = os.listdir("jsons/")
my_dicts = []
chunk_id = 0
for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    for chunk in content['chunks']:
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = create_embedding(chunk['text'])
        chunk_id += 1
        my_dicts.append(chunk)
        print(chunk)
    break


print(my_dicts)