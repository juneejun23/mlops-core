FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY result-worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY result-worker/ .

CMD ["python", "-u", "worker.py"]