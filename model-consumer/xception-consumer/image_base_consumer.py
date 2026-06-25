import os
import json
import time
import asyncio
import logging
import sys

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import tritonclient.grpc.aio as grpcclient
from prometheus_client import start_http_server

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_consumer import BaseConsumer, PENDING_REQUESTS

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class ImageBaseConsumer(BaseConsumer):
    def __init__(self, model_name):
        super().__init__(model_name)

    async def _process_message(self, msg_value, producer):
        task_id = msg_value.get('task_id')
        input_key = msg_value.get('input_key')
        num_image = msg_value.get('num_image')
        filename = os.path.basename(input_key)

        logger.info(f"Task {task_id} started - num image : {num_image} | input_key : {input_key}")

        server_ip = None
        tmp_path = None

        start_time = time.time()

        try:
            PENDING_REQUESTS.inc()
            loop = asyncio.get_running_loop()

            tmp_path = await loop.run_in_executor(None, self._download_s3_file, input_key)
            logger.info(f"Task {task_id}: S3 Downloaded in {time.time() - start_time:.2f}s")
            input_tensor = await loop.run_in_executor(None, self.preprocess, tmp_path)
            logger.info(f"Task {task_id}: Preprocessed in {time.time() - start_time:.2f}s")

            server_ip = await self._schedule_server()
            logger.info(f"Task {task_id}: Scheduled to {server_ip}")

            triton_client = grpcclient.InferenceServerClient(url=f"{server_ip}:8001")
            inputs = [grpcclient.InferInput("INPUT__0", input_tensor.shape, "FP32")]
            inputs[0].set_data_from_numpy(input_tensor)

            res = await triton_client.infer(model_name=self.model_name, inputs=inputs)
            output_data = res.as_numpy("OUTPUT__0")

            final_result = await loop.run_in_executor(None, self.postprocess, output_data)

            result_payload = {
                "task_id": task_id,
                "model_name": self.model_name,
                "filename": filename,
                "status": "SUCCESS",
                "result": final_result,
                "num_image": num_image,
                "error_msg": None,
                "timestamp": time.time()
            }
            await producer.send_and_wait(self.result_topic, json.dumps(result_payload).encode('utf-8'))
            logger.info(f"✅ [Done] Task: {task_id} | Server: {server_ip}")

        except Exception as e:
            logger.error(f"Task Failed: {task_id} | Error: {e}")

            error_str = str(e).lower()
            if server_ip and ("memory" in error_str or "oom" in error_str or "allocate" in error_str):
                logger.critical(f"🚨 OOM Detected on {server_ip}! Evicting model {self.model_name}...")
                try:
                    key = next((k for k in await self.redis.smembers("triton_servers") if server_ip in k), None)
                    if key:
                        triton_client = grpcclient.InferenceServerClient(url=f"{server_ip}:8001")
                        await triton_client.unload_model(self.model_name)
                        await self.redis.zrem(f"{key}:loaded_models", self.model_name)
                        await self._fetch_and_sync_vram(server_ip, key)
                except Exception as cleanup_err:
                    logger.error(f"Failed to evict model after OOM: {cleanup_err}")

            try:
                error_payload = {
                    "task_id": task_id,
                    "model_name": self.model_name,
                    "filename": filename,
                    "status": "FAILED",
                    "result": None,
                    "num_image": num_image,
                    "error_msg": str(e),
                    "timestamp": time.time()
                }
                await producer.send_and_wait(self.result_topic, json.dumps(error_payload).encode('utf-8'))
                logger.info(f"⚠️ [Reported Error] Task: {task_id} marked as FAILED in result topic.")
            except Exception as send_err:
                logger.error(f"Failed to send error report to Kafka: {send_err}")

            await asyncio.sleep(1.0)

        finally:
            self.semaphore.release()
            PENDING_REQUESTS.dec()

            if tmp_path:
                try:
                    os.remove(tmp_path)
                    logger.info(f"Removed: {tmp_path}")
                except Exception:
                    logger.warning(f"Not Removed: {tmp_path}")

            if server_ip:
                try:
                    key = next((k for k in await self.redis.smembers("triton_servers") if server_ip in k), None)
                    if key:
                        await self.redis.hincrby(f"{key}:stats", "load", -1)
                        logger.info(f"redis {key} decrement by 1")
                except Exception:
                    logger.warning(f"Redis Decrement {server_ip} passed")

    async def run(self):
        consumer = AIOKafkaConsumer(
            self.input_topic,
            bootstrap_servers=self.kafka_broker,
            group_id=f'{self.model_name}-group',
            auto_offset_reset='earliest',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        producer = AIOKafkaProducer(bootstrap_servers=self.kafka_broker)

        await consumer.start()
        await producer.start()

        self.semaphore = asyncio.Semaphore(self.local_max_tasks)

        start_http_server(self.metrics_port)
        logger.info(f"🚀 Started {self.model_name}. Local: {self.local_max_tasks}, Global: {self.global_queue_limit}")

        try:
            while True:
                await self.semaphore.acquire()

                if await self._is_cluster_saturated():
                    self.semaphore.release()
                    await asyncio.sleep(1)
                    continue

                try:
                    msg_set = await consumer.getmany(timeout_ms=1000, max_records=1)

                    if not msg_set:
                        self.semaphore.release()
                        continue

                    for tp, messages in msg_set.items():
                        for msg in messages:
                            logger.info(f"📨 [Kafka] 메시지 수신: topic={tp.topic}, partition={tp.partition}, offset={msg.offset}, value={msg.value}")
                            asyncio.create_task(self._process_message(msg.value, producer))

                except Exception as e:
                    logger.error(f"Loop error: {e}")
                    self.semaphore.release()
                    await asyncio.sleep(1)

        finally:
            await consumer.stop()
            await producer.stop()

    def start(self):
        asyncio.run(self.run())