import os
import json
import logging
from kafka import KafkaConsumer, KafkaProducer
import boto3
from botocore.client import Config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from ray.job_submission import JobSubmissionClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "my-kafka-cluster-kafka-bootstrap.kafka.svc.cluster.local:9092")
PREPROCESS_TOPIC = "preprocess-topic"
TRAIN_TOPIC = "train-topic"

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio.minio.svc.cluster.local:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "quHCnPBfDaYU0UsV0vfM")
TRAINING_BUCKET = os.environ.get("TRAINING_BUCKET", "training-data")

RAY_DASHBOARD_URL = os.environ.get("RAY_DASHBOARD_URL", "http://ray-cluster-kuberay-head-svc.kuberay.svc.cluster.local:8265")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://mlops:dashlove@postgres-cluster-rw.mlops-backend.svc.cluster.local:5432/mlops")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


def update_training_job_status(training_job_id: str, status: str, error_msg: str = None):
    db = SessionLocal()
    try:
        if error_msg:
            db.execute(text(
                "UPDATE training_job SET status=:status, error_msg=:error_msg, updated_at=:updated_at WHERE training_job_id=:id"
            ), {"status": status, "error_msg": error_msg, "updated_at": datetime.utcnow(), "id": training_job_id})
        else:
            db.execute(text(
                "UPDATE training_job SET status=:status, updated_at=:updated_at WHERE training_job_id=:id"
            ), {"status": status, "updated_at": datetime.utcnow(), "id": training_job_id})
        db.commit()
        logging.info(f"[DB] {training_job_id} → {status}")
    finally:
        db.close()


def get_training_job(training_job_id: str):
    db = SessionLocal()
    try:
        result = db.execute(text(
            "SELECT tenant_id, architecture FROM training_job WHERE training_job_id=:id"
        ), {"id": training_job_id}).fetchone()
        return result
    finally:
        db.close()


class BasePreprocessConsumer:

    def __init__(self):
        self.s3 = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
        self.consumer = KafkaConsumer(
            PREPROCESS_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            group_id="preprocess-consumer-group",
        )
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        self.ray_client = JobSubmissionClient(RAY_DASHBOARD_URL)
        logging.info(f"Connected to Ray dashboard: {RAY_DASHBOARD_URL}")

    def run(self):
        logging.info("Preprocess consumer started. Listening on preprocess-topic...")
        for message in self.consumer:
            payload = message.value
            training_job_id = payload["training_job_id"]
            logging.info(f"Received: {training_job_id}")
            try:
                update_training_job_status(training_job_id, "RUNNING")
                self._process(training_job_id, payload)
            except Exception as e:
                logging.error(f"Failed: {training_job_id} | {e}")
                update_training_job_status(training_job_id, "FAILED", error_msg=str(e))

    def _process(self, training_job_id, payload):
        job = get_training_job(training_job_id)
        tenant_id = job.tenant_id

        # payload에 tenant_id 추가
        job_payload = {
            "training_job_id": training_job_id,
            "zip_path": payload["zip_path"],
            "tenant_id": tenant_id,
            "face_based": payload.get("face_based", False),
        }

        # Ray Job 제출
        logging.info(f"Submitting Ray Job for {training_job_id}")
        job_id = self.ray_client.submit_job(
            entrypoint=f"python preprocess_job.py --payload '{json.dumps(job_payload)}'",
            runtime_env={
                "working_dir": "/app",
                "pip": ["facenet-pytorch==2.6.0", "boto3==1.43.45", "Pillow", "numpy==1.26.0"]
            }
        )
        logging.info(f"Ray Job submitted: {job_id}")

        # Job 완료 대기
        import time
        from ray.job_submission import JobStatus
        while True:
            status = self.ray_client.get_job_status(job_id)
            logging.info(f"Ray Job status: {status}")
            if status == JobStatus.SUCCEEDED:
                break
            elif status in (JobStatus.FAILED, JobStatus.STOPPED):
                logs = self.ray_client.get_job_logs(job_id)
                raise Exception(f"Ray Job failed: {logs[-500:]}")
            time.sleep(5)

        # 결과 파싱 (stdout에서 RESULT: 라인 찾기)
        logs = self.ray_client.get_job_logs(job_id)
        result_line = [l for l in logs.split("\n") if l.startswith("RESULT:")]
        if not result_line:
            raise Exception("No result found in Ray Job output")
        result = json.loads(result_line[-1].replace("RESULT:", ""))

        real_keys = result["real_keys"]
        fake_keys = result["fake_keys"]
        skipped = result["skipped"]

        logging.info(f"Cropped: real={len(real_keys)}, fake={len(fake_keys)}, skipped={skipped}")

        # train-topic 발행
        train_payload = {
            "training_job_id": training_job_id,
            "architecture": payload["architecture"],
            "epochs": payload["epochs"],
            "batch_size": payload["batch_size"],
            "lr": payload["lr"],
            "real_keys": real_keys,
            "fake_keys": fake_keys,
        }
        self.producer.send(TRAIN_TOPIC, train_payload)
        self.producer.flush()
        logging.info(f"Sent to train-topic: {training_job_id}")

        # DB COMPLETED 업데이트는 train-consumer가 담당
        # 여기서는 전처리 완료 후 train-topic 발행까지만


if __name__ == "__main__":
    consumer = BasePreprocessConsumer()
    consumer.run()