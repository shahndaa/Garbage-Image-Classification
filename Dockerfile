# Lightweight image that only serves the Gradio demo (no training dependencies).
FROM python:3.11-slim

WORKDIR /app

# System deps needed by Pillow/TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src/ src/
COPY app/ app/
COPY models/ models/

# Render sets $PORT at runtime; app.py reads it (defaults to 7860 locally)
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app/app.py"]
