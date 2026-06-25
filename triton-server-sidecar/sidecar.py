import os
import time
import socket
import signal
import sys
import redis

REDIS_HOST = os.environ.get("MLOPS_REDIS_HOST", "redis.mlops-backend.svc.cluster.local")
REDIS_PORT = int(os.environ.get("MLOPS_REDIS_PORT", 6379))
TRITON_PORT = int(os.environ.get("TRITON_PORT", 8000))
HEARTBEAT_INTERVAL = 5
DRAIN_TIMEOUT = 60

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def get_pod_ip():
    return socket.gethostbyname(socket.gethostname())


def server_key(ip):
    return f"server:{ip}:{TRITON_PORT}"


def start():
    ip = get_pod_ip()
    key = server_key(ip)

    print(f"Registering server {key} to Redis...")

    r.sadd("triton_servers", key)
    r.hset(f"{key}:stats", mapping={
        "ip": ip,
        "port": TRITON_PORT,
        "state": "ACTIVE",
        "load": 0,
        "last_heartbeat": time.time(),
        "loading_target": "None",
        "loading_started_at": 0,
        "total_vram": 24576,
        "used_vram": 0,
    })

    print(f"Registered. Sending heartbeat every {HEARTBEAT_INTERVAL}s...")

    running = True

    def handle_sigterm(signum, frame):
        nonlocal running
        print("SIGTERM received, stopping heartbeat loop...")
        running = False

    signal.signal(signal.SIGTERM, handle_sigterm)

    while running:
        r.hset(f"{key}:stats", "last_heartbeat", time.time())
        time.sleep(HEARTBEAT_INTERVAL)

    stop(ip)


def stop(ip=None):
    if ip is None:
        ip = get_pod_ip()
    key = server_key(ip)

    print(f"Draining {key}...")
    r.hset(f"{key}:stats", "state", "DRAINING")

    drain_start = time.time()
    while time.time() - drain_start < DRAIN_TIMEOUT:
        load = int(r.hget(f"{key}:stats", "load") or 0)
        if load == 0:
            break
        time.sleep(1)

    print(f"Removing {key} from Redis...")
    r.srem("triton_servers", key)
    r.delete(f"{key}:stats")
    r.delete(f"{key}:loaded_models")
    print("Done.")


if __name__ == "__main__":
    start()