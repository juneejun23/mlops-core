FROM python:3.11-slim

WORKDIR /app

COPY api-server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api-server/ .
COPY trainer/base/train_job.py .
COPY trainer/models/ models/
COPY trainer/configs/ configs/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]