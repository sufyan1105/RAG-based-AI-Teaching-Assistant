FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY process_incomming.py embeddings.joblib ./
CMD ["python", "process_incomming.py"]
