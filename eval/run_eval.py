"""Retrieval + answer evaluation. Run from repo root: python eval/run_eval.py
hit@3: any of the top-3 retrieved chunks is from the gold video and starts inside a gold time window.
Writes eval/results.jsonl (question, retrieved chunks, answer) for manual answer grading."""
import json
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from process_incomming import EMBEDDINGS_PATH, answer, to_mmss  # noqa: E402


def is_hit(chunks, video, gold):
    return any(
        int(r.number) == video and any(s <= r.start <= e for s, e in gold)
        for r in chunks.itertuples()
    )


if __name__ == "__main__":
    df = joblib.load(EMBEDDINGS_PATH)
    questions = json.load(open("eval/questions.json"))
    hits = 0
    with open("eval/results.jsonl", "w") as out:
        for i, item in enumerate(questions, 1):
            resp, chunks = answer(df, item["q"])
            hit = is_hit(chunks.head(3), item["video"], item["gold"])
            hits += hit
            out.write(json.dumps({
                "id": i, "q": item["q"], "video": item["video"], "gold": item["gold"], "hit@3": hit,
                "retrieved": [f"#{r.number} {to_mmss(r.start)}-{to_mmss(r.end)}: {r.text.strip()}" for r in chunks.itertuples()],
                "answer": resp,
            }) + "\n")
            print(f"{i:2d} hit@3={int(hit)} {item['q']}")
    print(f"\nhit@3 = {hits}/{len(questions)} = {hits / len(questions):.0%}")
