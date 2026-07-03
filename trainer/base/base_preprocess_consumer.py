# trainer/base/base_preprocess_consumer.py
import os
import json
import zipfile
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from kafka import KafkaConsumer, KafkaProducer
import boto3
from botocore.client import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "my-kafka-cluster-kafka-bootstrap.kafka.svc.cluster.local:9092")
PREPROCESS_TOPIC = "preprocess-topic"
TRAIN_TOPIC = "train-topic"

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio.minio.svc.cluster.local:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "quHCnPBfDaYU0UsV0vfM")
TRAINING_BUCKET = os.environ.get("TRAINING_BUCKET", "training-data")


class BasePreprocessConsumer(ABC):

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

    @abstractmethod
    def preprocess(self, image_path: str) -> bytes:
        """
        이미지 파일을 읽어서 전처리 후 numpy array를 bytes로 반환.
        모델별로 오버라이드해야 함.
        """
        pass

    def run(self):
        logging.info("Preprocess consumer started. Listening on preprocess-topic...")
        for message in self.consumer:
            payload = message.value
            training_job_id = payload["training_job_id"]
            zip_path = payload["zip_path"]
            logging.info(f"Received: {training_job_id}")

            try:
                self._process(training_job_id, zip_path, payload)
            except Exception as e:
                logging.error(f"Failed: {training_job_id} | {e}")

    def _process(self, training_job_id, zip_path, payload):
        work_dir = f"/tmp/{training_job_id}"
        zip_local = f"{work_dir}/upload.zip"
        extract_dir = f"{work_dir}/data"

        # 1. zip 다운로드
        logging.info(f"[1/4] Downloading zip: {zip_path}")
        os.makedirs(work_dir, exist_ok=True)
        self.s3.download_file(TRAINING_BUCKET, zip_path, zip_local)

        # 2. 압축 해제
        logging.info(f"[2/4] Extracting zip")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_local, "r") as zf:
            zf.extractall(extract_dir)

        # 3. labels.json 파싱
        logging.info(f"[3/4] Parsing labels.json")
        labels_path = Path(extract_dir) / "labels.json"
        if not labels_path.exists():
            raise FileNotFoundError("labels.json not found")
        with open(labels_path) as f:
            labels = json.load(f)

        # 4. 전처리 + MinIO 업로드
        logging.info(f"[4/4] Preprocessing and uploading")
        processed = {"real": [], "fake": [], "skipped": 0}

        all_images = list(Path(extract_dir).rglob("*.jpg")) + \
                     list(Path(extract_dir).rglob("*.png")) + \
                     list(Path(extract_dir).rglob("*.jpeg"))

        for img_path in all_images:
            filename = img_path.name
            if filename not in labels:
                processed["skipped"] += 1
                continue

            label = labels[filename]
            tensor_bytes = self.preprocess(str(img_path))

            # MinIO에 전처리 결과 저장
            object_key = f"tenants/preprocessed/{training_job_id}/{label}/{filename}.npy"
            self.s3.put_object(
                Bucket=TRAINING_BUCKET,
                Key=object_key,
                Body=tensor_bytes,
            )
            processed[label].append(object_key)

        logging.info(f"Preprocessed: real={len(processed['real'])}, fake={len(processed['fake'])}, skipped={processed['skipped']}")

        # 5. train-topic으로 발행
        train_payload = {
            "training_job_id": training_job_id,
            "architecture": payload["architecture"],
            "epochs": payload["epochs"],
            "batch_size": payload["batch_size"],
            "lr": payload["lr"],
            "real_keys": processed["real"],
            "fake_keys": processed["fake"],
        }
        self.producer.send(TRAIN_TOPIC, train_payload)
        self.producer.flush()
        logging.info(f"Sent to train-topic: {training_job_id}")