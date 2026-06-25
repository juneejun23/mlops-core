import time
import json
from sqlalchemy import create_engine, Column, String, Integer, JSON, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from kafka import KafkaProducer

DATABASE_URL = "postgresql://mlops:dashlove@postgres-cluster-rw.mlops-backend.svc.cluster.local:5432/mlops"
KAFKA_BOOTSTRAP_SERVERS = "my-kafka-cluster-kafka-bootstrap.kafka.svc.cluster.local:9092"
POLL_INTERVAL = 5

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Outbox(Base):
    __tablename__ = "outbox"

    outbox_id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, nullable=False)
    model_id = Column(String, nullable=False)
    type = Column(String, nullable=False)
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime)


producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    key_serializer=lambda k: str(k).encode("utf-8"),
)


def process_outbox_events():
    db = SessionLocal()
    try:
        events = db.query(Outbox).order_by(Outbox.created_at.asc()).limit(100).all()
        for event in events:
            topic = f"input-topic-{event.model_id}"
            producer.send(topic, key=event.outbox_id, value=event.payload)
            producer.flush()
            db.delete(event)
            db.commit()
            print(f"Published outbox_id={event.outbox_id} to topic={topic}")
    finally:
        db.close()


if __name__ == "__main__":
    print("Kafka Publisher started. Polling every", POLL_INTERVAL, "seconds...")
    while True:
        process_outbox_events()
        time.sleep(POLL_INTERVAL)