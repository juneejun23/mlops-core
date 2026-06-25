import os
import time
import asyncio
import logging
import numpy as np
import boto3
import tempfile
import aiohttp
import sys
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import redis.asyncio as redis
import tritonclient.grpc.aio as grpcclient
from prometheus_client import Gauge

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from abstract_consumer import AbstractConsumer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PENDING_REQUESTS = Gauge('consumer_pending_requests', 'Number of active tasks in consumer memory')


class BaseConsumer(AbstractConsumer):
    def __init__(self, model_name):
        self.model_name = model_name

        self.kafka_broker = os.environ.get("KAFKA_BROKER", "my-kafka-cluster-kafka-bootstrap.kafka.svc.cluster.local:9092")
        self.input_topic = os.environ.get("INPUT_TOPIC", f"input-topic-{model_name}")
        self.result_topic = os.environ.get("RESULT_TOPIC", "inference-result-topic")
        self.group_id = os.environ.get("CONSUMER_GROUP", f"{model_name}-group")

        self.redis_host = os.environ.get("MLOPS_REDIS_HOST", "redis.mlops-backend.svc.cluster.local")
        self.redis_port = int(os.environ.get("MLOPS_REDIS_PORT", 6379))
        self.metrics_port = int(os.environ.get("METRICS_PORT", 8080))

        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL", "http://minio.minio.svc.cluster.local:9000"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "quHCnPBfDaYU0UsV0vfM"),
            region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
        self.bucket_name = os.environ.get("S3_BUCKET_NAME", "hyjk826-mlops-1011")

        self.req_vram = None
        self.TRITON_LOAD_TIMEOUT = 600
        self.SCHEDULE_TIMEOUT = 600

        self.local_max_tasks = 100
        self.global_queue_limit = 512

        self.redis = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=True)
        self.semaphore = None

        self.COST_LOADING_WAIT = 10
        self.COST_NEW_LOAD = 30

        self.lock_script = self.redis.register_script("""
            local current_state = redis.call('HGET', KEYS[1], 'state')
            if current_state == 'ACTIVE' then
                redis.call('HMSET', KEYS[1], 'state', 'DRAINING', 'loading_target', ARGV[1])
                return 1
            else
                return 0
            end
        """)

    def softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        return exp_a / sum_exp_a

    def _download_s3_file(self, input_key):
        file_ext = os.path.splitext(input_key)[1]
        if not file_ext:
            file_ext = ".bin"

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        tmp_path = tmp_file.name
        tmp_file.close()

        try:
            self.s3_client.download_file(self.bucket_name, input_key, tmp_path)
            return tmp_path
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise e

    async def _is_cluster_saturated(self):
        try:
            return not bool(await self.redis.smembers("triton_servers"))
        except Exception:
            return True

    def _calculate_cost(self, server_stats):
        state = server_stats.get('state')
        current_load = int(server_stats.get('load', 0))
        loading_target = server_stats.get('loading_target')
        used_vram = int(server_stats.get('used_vram', 0))
        model_count = server_stats.get("model_count", 0)
        is_my_model = server_stats.get('is_my_model', False)

        if state == 'ACTIVE' and is_my_model:
            return current_load

        if state in ['LOADING', 'DRAINING'] and loading_target == self.model_name:
            return current_load + self.COST_LOADING_WAIT

        if state == 'ACTIVE' and model_count == 0:
            return self.COST_NEW_LOAD

        if state == 'ACTIVE' and not is_my_model:
            return current_load + self.COST_NEW_LOAD

        return 999999

    async def _fetch_and_sync_vram(self, server_ip, server_key):
        metrics_url = f"http://{server_ip}:8002/metrics"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(metrics_url, timeout=2) as response:
                    if response.status != 200:
                        logger.error(f"Failed to sync VRAM from {server_ip}: HTTP {response.status}")
                        return None

                    text = await response.text()

                    for line in text.split('\n'):
                        if line.startswith("nv_gpu_memory_used_bytes") and 'state="used"' in line:
                            used_bytes = float(line.split(' ')[-1])
                            used_mb = int(used_bytes / (1024 * 1024))
                            await self.redis.hset(f"{server_key}:stats", "used_vram", used_mb)
                            return used_mb

            logger.warning(f"Metric not found in {server_ip}")
            return None

        except Exception as e:
            logger.error(f"Error syncing VRAM from {server_ip}: {e}")
            return None

    async def _try_lock_and_load(self, target_server):
        logger.info(f"try lock and load {target_server} with {self.model_name}")
        success = await self.lock_script(keys=[f"{target_server['key']}:stats"], args=[self.model_name])
        if not success:
            return False

        await self.redis.hset(f"{target_server['key']}:stats", "loading_started_at", time.time())

        try:
            drain_start = time.time()
            logger.info(f"[Drain] Started on {target_server['ip']} for model '{self.model_name}'")
            last_warn = 0
            while True:
                load = int(await self.redis.hget(f"{target_server['key']}:stats", "load") or 0)
                if load == 0:
                    break
                elapsed = time.time() - drain_start
                if elapsed > 5 and elapsed - last_warn >= 5:
                    logger.warning(f"⚠️ [Drain] {target_server['ip']} stuck for {elapsed:.1f}s. Current load: {load}")
                    last_warn = elapsed
                await asyncio.sleep(0.1)
            logger.info(f"[Drain] Completed on {target_server['ip']} in {time.time() - drain_start:.1f}s")

            async def _unsafe_loading_process():
                await self.redis.hset(f"{target_server['key']}:stats", "state", "LOADING")
                triton_client = grpcclient.InferenceServerClient(url=f"{target_server['ip']}:8001")

                total_vram = int(await self.redis.hget(f"{target_server['key']}:stats", "total_vram") or 0)
                current_used_vram = int(await self.redis.hget(f"{target_server['key']}:stats", "used_vram") or 0)

                while (total_vram - current_used_vram) < self.req_vram:
                    lru_list = await self.redis.zrange(f"{target_server['key']}:loaded_models", 0, 0)
                    if not lru_list:
                        break

                    victim_model = lru_list[0]
                    await triton_client.unload_model(victim_model)
                    await self.redis.zrem(f"{target_server['key']}:loaded_models", victim_model)

                    synced_vram = await self._fetch_and_sync_vram(target_server['ip'], target_server['key'])
                    if synced_vram is not None:
                        current_used_vram = synced_vram
                    else:
                        await asyncio.sleep(1)

                await triton_client.load_model(self.model_name)

                final_vram = await self._fetch_and_sync_vram(target_server['ip'], target_server['key'])
                if final_vram is None:
                    final_vram = current_used_vram + 2000
                    await self.redis.hset(f"{target_server['key']}:stats", "used_vram", final_vram)
                return True

            load_start = time.time()
            await asyncio.wait_for(_unsafe_loading_process(), timeout=self.TRITON_LOAD_TIMEOUT)
            logger.info(f"[Load] Model '{self.model_name}' loaded on {target_server['ip']} in {time.time() - load_start:.1f}s")

            await self.redis.hset(f"{target_server['key']}:stats", mapping={
                "state": "ACTIVE",
                "loading_target": "None",
                "loading_started_at": 0
            })

            await self.redis.zadd(f"{target_server['key']}:loaded_models", {self.model_name: time.time()})
            await self.redis.hincrby(f"{target_server['key']}:stats", "load", 1)

            return True

        except asyncio.TimeoutError:
            logger.error(f"⏱️ Load Timeout ({self.TRITON_LOAD_TIMEOUT}s) on {target_server['ip']}. Rolling back...")
            await self._rollback_state(target_server)
            return False

        except Exception as e:
            logger.error(f"Load failed on {target_server['ip']}: {e}")
            await self._rollback_state(target_server)
            return False

    async def _rollback_state(self, target_server):
        try:
            await self.redis.hset(f"{target_server['key']}:stats", mapping={
                "state": "ACTIVE",
                "loading_target": "None",
                "loading_started_at": 0
            })
            await self._fetch_and_sync_vram(target_server['ip'], target_server['key'])
        except Exception as e:
            logger.error(f"Rollback failed: {e}")

    async def _schedule_server(self):
        schedule_start = time.time()

        while True:
            elapsed = time.time() - schedule_start
            if elapsed > self.SCHEDULE_TIMEOUT:
                raise TimeoutError(
                    f"서버 스케줄링 타임아웃: {elapsed:.0f}초 동안 사용 가능한 Triton 서버 없음"
                )

            server_keys = await self.redis.smembers("triton_servers")
            if not server_keys:
                logger.warning(f"[Schedule] 등록된 서버 없음. 대기 중... ({elapsed:.0f}s)")
                await asyncio.sleep(1)
                continue

            candidates = []

            for key in server_keys:
                stats = await self.redis.hgetall(f"{key}:stats")

                if not stats or stats.get('state') == 'DOWN':
                    continue

                stats['ip'] = stats['ip']
                stats['key'] = key
                stats['model_count'] = await self.redis.zcard(f"{key}:loaded_models")
                stats['is_my_model'] = await self.redis.zscore(f"{key}:loaded_models", self.model_name) is not None

                score = self._calculate_cost(stats)
                candidates.append((score, stats))

                logger.info(f"[DONE] Scanning {key}")

            if not candidates:
                await asyncio.sleep(1)
                continue

            best_score, target = min(candidates, key=lambda x: x[0])

            if best_score > self.global_queue_limit:
                logger.info(f"Expensive. Wait")
                await asyncio.sleep(1)
                continue

            if target['state'] == 'ACTIVE' and target['is_my_model']:
                await self.redis.hincrby(f"{target['key']}:stats", "load", 1)
                await self.redis.zadd(f"{target['key']}:loaded_models", {self.model_name: time.time()})
                return target['ip']

            if target['state'] in ['LOADING', 'DRAINING'] and target['loading_target'] == self.model_name:
                loading_started_at = float(target.get('loading_started_at', 0))
                elapsed = time.time() - loading_started_at

                if loading_started_at > 0 and elapsed > (self.TRITON_LOAD_TIMEOUT * 1.2):
                    logger.warning(f"🧟 Zombie State Detected on {target['ip']}! (Elapsed: {elapsed:.1f}s). Force Resetting...")
                    await self.redis.hset(f"{target['key']}:stats", mapping={
                        "state": "ACTIVE",
                        "loading_target": "None",
                        "loading_started_at": 0
                    })

                await asyncio.sleep(0.5)
                continue

            if target['state'] == 'ACTIVE' and not target['is_my_model']:
                if await self._try_lock_and_load(target):
                    return target['ip']
                else:
                    continue

            await asyncio.sleep(0.1)