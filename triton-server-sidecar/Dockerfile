FROM python:3.12-slim

WORKDIR /app

COPY triton-server-sidecar/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY triton-server-sidecar/ .

CMD ["python", "-u", "sidecar.py"]