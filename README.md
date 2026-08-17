# 🎓 RAG-Based AI Teaching Assistant

A Retrieval-Augmented Generation (RAG) system that acts as an AI teaching assistant for a web development course. Students can ask questions in natural language and the assistant tells them **which video** covers the topic and **at what timestamp** — so they can jump straight to the relevant part.

---

## 💡 How It Works

The pipeline takes raw course videos all the way to a queryable AI assistant in 5 steps:

```
Videos → MP3s → JSON Transcripts → Embeddings → LLM Response
```

### Step 1 — Collect your videos
Place all your video files into the `videos/` folder.

### Step 2 — Convert videos to MP3 (`video_to_mp3.py`)
Converts all video files in the `videos/` folder to MP3 audio files.

### Step 3 — Transcribe MP3s to JSON (`mp3_to_json.py`)
Converts each MP3 to a chunked JSON transcript with text and timestamps.

### Step 4 — Build the vector store (`preprocess_json.py`)
Reads all JSON files from the `jsons/` folder, generates **bge-m3** embeddings for each chunk via Ollama, and saves everything as a `pandas` DataFrame in `embeddings.joblib`.

### Step 5 — Query the assistant (`process_incomming.py`)
Loads `embeddings.joblib`, embeds the student's question using **bge-m3**, finds the top 5 most similar chunks via cosine similarity, and feeds them to **Llama 3.2** with a prompt that directs the student to the right video and timestamp. The response is printed and saved to `response.txt`.

---

## 📁 Project Structure

```
RAG-based-AI-Teaching-Assistant/
│
├── video_to_mp3.py          # Step 2: Convert video files to MP3
├── mp3_to_json.py           # Step 3: Transcribe MP3s to chunked JSON
├── preprocess_json.py       # Step 4: Embed JSON chunks → embeddings.joblib
├── process_incomming.py     # Step 5: Query handler (RAG + LLM)
│
├── embeddings.joblib        # Pre-built vector store
├── response.txt             # Last generated assistant response
│
├── videos/                  # (Expected) Raw course video files
├── jsons/                   # (Expected) Chunked transcript JSON files
└── Unused py files/         # Experimental/draft scripts
```

---

## 🛠️ Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running locally

### Pull the required models

```bash
ollama pull bge-m3
ollama pull llama3.2
```

### Install Python dependencies

```bash
pip install numpy pandas scikit-learn joblib requests
```

---

## 🚀 Usage

```bash
# Step 2 - Convert videos to MP3
python video_to_mp3.py

# Step 3 - Transcribe MP3s to JSON
python mp3_to_json.py

# Step 4 - Build the vector store
python preprocess_json.py

# Step 5 - Ask a question
python process_incomming.py
```

Example interaction:

```
Ask your question: Where is CSS Grid explained?

Response: CSS Grid is covered in video "05_CSS Grid Basics" starting at 00:03:20.
```

Steps 2–4 only need to be run once. After `embeddings.joblib` is built, you can run Step 5 repeatedly.

---

## 🔧 Tech Stack

| Component | Tool |
|---|---|
| Embedding Model | bge-m3 (via Ollama) |
| Language Model | Llama 3.2 (via Ollama) |
| Similarity Search | Cosine Similarity (scikit-learn) |
| Vector Store | pandas DataFrame + joblib |
| Inference Runtime | Ollama (local) |

---

## 👤 Author

**Sufyan** — [github.com/sufyan1105](https://github.com/sufyan1105)
