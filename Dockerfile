FROM python:3.14-slim

WORKDIR /app

COPY api-server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api-server/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]