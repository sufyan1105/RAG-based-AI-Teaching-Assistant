# In this we are going to use similar code as speech_to_text.py to convert all the audio files to text files.

import whisper 
import json 
import os

model = whisper.load_model("large-v2")
audios = os.listdir("audios/")


for audio in audios:
    if("_" in audio):
        number = audio.split("_")[0]
        name = audio.split("_")[1].split(".mp3")[0]
        results = model.transcribe(audio = f"audios/{audio}", language = "hi", task = "translate", word_timestamps = False)


        chunks = []
        for segment in results["segments"]:
            chunks.append({"number": number,"name": name, "start": segment["start"], "end": segment["end"], "text": segment["text"]})

        chunks_with_metadata = {"chunks": chunks, "text": results["text"]}

        with open(f"jsons/{number}_{name}.json", "w") as f:
            json.dump(chunks_with_metadata, f,)