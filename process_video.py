# Converts the videos to MP3 format and saves them in the audios folder using ffmpeg.

import os 
import subprocess

files = os.listdir("videos/")

# print(files)

for file in files:
    # print(file)
    tutorial_number = file.split(" [")[0].split(" #")[1]
    file_name = file.split(" ｜ ")[0]
    print(tutorial_number, file_name)
    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{tutorial_number}_{file_name}.mp3"]) 

