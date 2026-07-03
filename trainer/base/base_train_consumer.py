# trainer/base/base_train_consumer.py
import os
import json
import logging
from abc import ABC, abstractmethod
from kafka import KafkaConsumer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "my-kafka-cluster-kafka-bootstrap.kafka.svc.cluster.local:9092")
TRAIN_TOPIC = "train-topic"


class BaseTrainConsumer(ABC):

    def __init__(self):
        self.consumer = KafkaConsumer(
            TRAIN_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            group_id="train-consumer-group",
        )

    @abstractmethod
    def train(self, payload: dict):
        """
        실제 학습 로직. 모델별로 오버라이드해야 함.
        """
        pass

    def run(self):
        logging.info("Train consumer started. Listening on train-topic...")
        for message in self.consumer:
            payload = message.value
            training_job_id = payload["training_job_id"]
            logging.info(f"Received: {training_job_id}")

            try:
                self.train(payload)
            except Exception as e:
                logging.error(f"Failed: {training_job_id} | {e}")