# In this we are going to use whisper to convert the audio files to text files.

import whisper 
import json 

model = whisper.load_model("large-v2")

results = model.transcribe(audio = "audios/2_Your First HTML Website.mp3", language = "hi", task = "translate")
print(results["segments"])

chunks = []
for segment in results["segments"]:
    chunks.append({"start": segment["start"], "end": segment["end"], "text": segment["text"]})

print(chunks)

with open("output.json", "w") as f:
    json.dump(chunks, f,)


