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

def inference(prompt):
    r = requests.post(
          "http://localhost:11434/api/generate",
          json={"model": "llama3.2",
                "prompt": prompt,
                "stream": False,}
    )
    response = r.json()
    print(response)
    return response

if __name__ == "__main__":
  df = joblib.load("embeddings.joblib")

  incoming_query = input("Ask your question: ")
  question_embedding = create_embedding([incoming_query])[0]

  similarities = cosine_similarity(
      np.vstack(df["embedding"].values), [question_embedding]
  ).flatten()
  top_results = 5
  max_indx = similarities.argsort()[::-1][0:top_results]
#   print(f"Similarities: {max_indx}")
  new_df = df.loc[max_indx]
#   print(new_df[["name", "text", "number"]])

prompt = f''' I am teaching web development course. Here are video chunks containing video names, text, number, start_time , end_time an the text at that time .:
{new_df[["name", "text", "number", "start", "end"]].to_json(orient="records")}
-----------------------------------------------------------
" {incoming_query} "
User asked this question related to the video chunks, you have to answer where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that part. If user asks unrelated questions tell him you can only answer questions related to the Course. 
'''

response = (inference(prompt))["response"]
print(f"Response: {response}")

with open("response.txt", "w") as f:
    f.write(response)
# for index, row in new_df.iterrows():
#     print(f"Name: {row['name']}")
#     print(f"Text: {row['text']}")
#     print(f"Number: {row['number']}")
#     print(f"start_time: {row['start']}")
#     print(f"end_time: {row['end']}")
#     print("--------------------------------------------------")