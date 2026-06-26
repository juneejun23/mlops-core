import os
import json
import time
import redis
import boto3
from kafka import KafkaConsumer
from sqlalchemy import create_engine, Column, String, Integer, JSON, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "postgresql://mlops:dashlove@postgres-cluster-rw.mlops-backend.svc.cluster.local:5432/mlops"
KAFKA_BOOTSTRAP_SERVERS = "my-kafka-cluster-kafka-bootstrap.kafka.svc.cluster.local:9092"
RESULT_TOPIC = "inference-result-topic"

MLOPS_REDIS_HOST = os.environ.get("MLOPS_REDIS_HOST", "redis.mlops-backend.svc.cluster.local")
MLOPS_REDIS_PORT = int(os.environ.get("MLOPS_REDIS_PORT", 6379))

REDIS_RESULT_TTL = 3 * 24 * 60 * 60  # 3일

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Task(Base):
    __tablename__ = "task"

    task_id = Column(String, primary_key=True)
    tenant_id = Column(String, nullable=False)
    model_id = Column(String, nullable=False)
    num_image = Column(Integer, nullable=False)
    status = Column(String, nullable=False, default="PENDING")
    result_url = Column(String, default="")
    meta = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


r = redis.Redis(host=MLOPS_REDIS_HOST, port=MLOPS_REDIS_PORT, decode_responses=True)

s3_client = boto3.client(
    's3',
    endpoint_url=os.environ.get("AWS_ENDPOINT_URL", "http://minio.minio.svc.cluster.local:9000"),
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minio"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "quHCnPBfDaYU0UsV0vfM"),
    region_name="us-east-1",
)
BUCKET_NAME = "hyjk826-mlops-1011"

consumer = KafkaConsumer(
    RESULT_TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    group_id="result-worker-group",
)


def get_task(task_id):
    session = SessionLocal()
    try:
        return session.query(Task).filter(Task.task_id == task_id).first()
    finally:
        session.close()


def update_task_status(task_id, status, result_url=None):
    session = SessionLocal()
    try:
        task = session.query(Task).filter(Task.task_id == task_id).first()
        if task:
            task.status = status
            if result_url is not None:
                task.result_url = result_url
            task.updated_at = datetime.utcnow()
            session.commit()
    finally:
        session.close()


def process_message(payload):
    task_id = payload["task_id"]
    print(f"Received result for task_id={task_id}, status={payload.get('status')}")

    redis_key = f"task:{task_id}:results"
    count_key = f"task:{task_id}:count"

    r.rpush(redis_key, json.dumps(payload))
    r.expire(redis_key, REDIS_RESULT_TTL)

    new_count = r.incr(count_key)
    r.expire(count_key, REDIS_RESULT_TTL)

    if new_count == 1:
        update_task_status(task_id, "RUNNING")
        print(f"[Task DB] {task_id} -> RUNNING")

    task = get_task(task_id)
    if task is None:
        print(f"Task {task_id} not found in DB")
        return

    if new_count >= task.num_image:
        all_results_raw = r.lrange(redis_key, 0, -1)
        all_results = [json.loads(item) for item in all_results_raw]

        final_payload = {
            "task_id": task_id,
            "num_image": task.num_image,
            "results": all_results,
        }

        object_key = f"results/{task_id}/result.json"
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=object_key,
            Body=json.dumps(final_payload).encode("utf-8"),
            ContentType="application/json",
        )

        result_url = f"{BUCKET_NAME}/{object_key}"
        update_task_status(task_id, "COMPLETED", result_url=result_url)
        print(f"[Task DB] {task_id} -> COMPLETED, result saved to {object_key}")


if __name__ == "__main__":
    print(f"result-worker started. Listening on topic: {RESULT_TOPIC}")
    for message in consumer:
        process_message(message.value)