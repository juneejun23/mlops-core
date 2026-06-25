FROM python:3.14-slim

WORKDIR /app

COPY kafka-publisher/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY kafka-publisher/ .

CMD ["python", "-u", "publisher.py"]