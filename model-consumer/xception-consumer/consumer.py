import json
from kafka import KafkaConsumer

KAFKA_BOOTSTRAP_SERVERS = "my-kafka-cluster-kafka-bootstrap.kafka.svc.cluster.local:9092"
TOPIC = "input-topic-xception"

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    group_id="xception-consumer-group",
)


def process_message(payload):
    print(f"Received message: {payload}")
    task_id = payload["task_id"]
    input_key = payload["input_key"]
    num_image = payload["num_image"]
    print(f"task_id={task_id}, input_key={input_key}, num_image={num_image}")


if __name__ == "__main__":
    print(f"xception-consumer started. Listening on topic: {TOPIC}")
    for message in consumer:
        process_message(message.value)