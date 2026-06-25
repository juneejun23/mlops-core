FROM python:3.14-slim

WORKDIR /app

COPY model-consumer/xception-consumer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model-consumer/xception-consumer/ .

CMD ["python", "-u", "consumer.py"]