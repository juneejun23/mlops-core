"""
train_job.py — Ray Train으로 xception 학습.
Ray Job API로 제출되어 클러스터 내부에서 실행됨.
"""
import os
import io
import sys
import json
import argparse
import tempfile
import logging
import yaml
from datetime import datetime

import numpy as np
from PIL import Image
import boto3
from botocore.client import Config
import torch
import torch.nn as nn
import ray
import ray.train
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.data import ActorPoolStrategy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio.minio.svc.cluster.local:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "quHCnPBfDaYU0UsV0vfM")
TRAINING_BUCKET = os.environ.get("TRAINING_BUCKET", "training-data")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://mlops:dashlove@postgres-cluster-rw.mlops-backend.svc.cluster.local:5432/mlops")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


def update_status(training_job_id: str, status: str, metrics: dict = None, error_msg: str = None):
    db = SessionLocal()
    try:
        if metrics:
            db.execute(text(
                "UPDATE training_job SET status=:status, metrics=:metrics, updated_at=:updated_at WHERE training_job_id=:id"
            ), {"status": status, "metrics": json.dumps(metrics), "updated_at": datetime.utcnow(), "id": training_job_id})
        elif error_msg:
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


def validate_architecture(architecture: str) -> dict:
    """모델명, yaml, .py 파일 일치 확인. 통과 시 config 반환."""
    config_path = f"configs/{architecture}.yaml"
    model_path = f"models/{architecture}.py"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(config_path) as f:
        model_config = yaml.safe_load(f)

    config_name = model_config.get("model", {}).get("name")
    if config_name != architecture:
        raise ValueError(
            f"Architecture mismatch: payload='{architecture}' but config says '{config_name}'"
        )

    logging.info(f"[train_job] 모델 검증 완료: {architecture}")
    print(f"STDOUT: 모델 검증 완료 — {architecture}")
    return model_config


class ImagePreprocessor:
    def __init__(self):
        self.s3 = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )

    def __call__(self, batch: dict) -> dict:
        images = []
        labels = []
        for i in range(len(batch["minio_path"])):
            minio_path = batch["minio_path"][i]
            label = batch["label"][i]

            response = self.s3.get_object(Bucket=TRAINING_BUCKET, Key=minio_path)
            img_bytes = response["Body"].read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            img = img.resize((256, 256))
            img_array = np.array(img).astype(np.float32)
            img_array = (img_array / 255.0 - 0.5) / 0.5
            img_array = np.transpose(img_array, (2, 0, 1))

            images.append(img_array)
            labels.append(0 if label == "real" else 1)

        return {
            "image": np.stack(images),
            "label": np.array(labels),
        }


class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, reps, stride=1, start_with_relu=True, grow_first=True):
        super().__init__()
        self.skip = None
        self.skip_bn = None
        if out_ch != in_ch or stride != 1:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_ch)

        rep = []
        filters = in_ch
        if grow_first:
            rep += [nn.ReLU(inplace=False), SeparableConv2d(in_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch)]
            filters = out_ch
        for _ in range(reps - 1):
            rep += [nn.ReLU(inplace=False), SeparableConv2d(filters, filters, 3, 1, 1), nn.BatchNorm2d(filters)]
        if not grow_first:
            rep += [nn.ReLU(inplace=False), SeparableConv2d(in_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch)]
        if not start_with_relu:
            rep = rep[1:]
        if stride != 1:
            rep += [nn.MaxPool2d(3, stride, 1)]
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)
        skip = x if self.skip is None else self.skip_bn(self.skip(x))
        return out + skip


class Xception(nn.Module):
    def __init__(self, num_classes=2, in_chans=3):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_chans, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, 2, stride=2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, stride=2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, stride=2, start_with_relu=True, grow_first=True)
        self.middle = nn.Sequential(*[Block(728, 728, 3, stride=1) for _ in range(8)])
        self.block12 = Block(728, 1024, 2, stride=2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.middle(x)
        x = self.block12(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def train_loop_per_worker(config: dict):
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    architecture = config["architecture"]
    num_classes = config["num_classes"]
    tenant_id = config["tenant_id"]
    training_job_id = config["training_job_id"]

    train_shard = train.get_dataset_shard("train")

    if architecture == "xception":
        model = Xception(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    model = ray.train.torch.prepare_model(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        correct = 0
        total = 0

        for batch in train_shard.iter_torch_batches(
            batch_size=batch_size,
            dtypes={"image": torch.float32, "label": torch.long}
        ):
            inputs = batch["image"]
            labels = batch["label"]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = correct / max(total, 1)

        print(f"[train_job] Epoch {epoch}: loss={avg_loss:.4f}, accuracy={accuracy:.4f}")
        ray.train.report({"epoch": epoch, "loss": avg_loss, "accuracy": accuracy})

        # 마지막 epoch에서 rank 0만 MinIO에 모델 저장
        if epoch == epochs - 1 and ray.train.get_context().get_world_rank() == 0:
            base_model = model.module if hasattr(model, "module") else model
            buf = io.BytesIO()
            torch.save(base_model.state_dict(), buf)
            buf.seek(0)

            s3 = boto3.client(
                "s3",
                endpoint_url=MINIO_ENDPOINT,
                aws_access_key_id=MINIO_ACCESS_KEY,
                aws_secret_access_key=MINIO_SECRET_KEY,
                config=Config(signature_version="s3v4"),
                region_name="us-east-1",
            )
            model_key = f"tenants/{tenant_id}/training-jobs/{training_job_id}/model/model.pt"
            s3.put_object(Bucket=TRAINING_BUCKET, Key=model_key, Body=buf.getvalue())
            print(f"[train_job] 모델 저장 완료: {model_key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=str, required=True)
    args = parser.parse_args()

    payload = json.loads(args.payload)
    training_job_id = payload["training_job_id"]
    real_keys = payload["real_keys"]
    fake_keys = payload["fake_keys"]
    epochs = payload["epochs"]
    batch_size = payload["batch_size"]
    lr = payload["lr"]
    tenant_id = payload["tenant_id"]
    architecture = payload["architecture"]

    # 모델명 + yaml + .py 파일 일치 확인
    model_config = validate_architecture(architecture)
    num_classes = model_config["model"]["num_classes"]

    ray.init()
    logging.info(f"[train_job] 학습 시작: {training_job_id}")
    print(f"STDOUT: 학습 시작 — {training_job_id}")

    try:
        update_status(training_job_id, "TRAINING")

        items = (
            [{"minio_path": k, "label": "real"} for k in real_keys] +
            [{"minio_path": k, "label": "fake"} for k in fake_keys]
        )
        ds = ray.data.from_items(items)
        ds = ds.map_batches(
            ImagePreprocessor,
            compute=ActorPoolStrategy(size=1),
            batch_size=16,
        )

        trainer = TorchTrainer(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config={
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "architecture": architecture,
                "num_classes": num_classes,
                "tenant_id": tenant_id,
                "training_job_id": training_job_id,
            },
            datasets={"train": ds},
            scaling_config=ScalingConfig(
                num_workers=1,
                use_gpu=True,
            ),
        )

        result = trainer.fit()
        final_loss = result.metrics.get("loss") if result.metrics else None
        final_accuracy = result.metrics.get("accuracy") if result.metrics else None

        logging.info(f"[train_job] 학습 완료: loss={final_loss}, accuracy={final_accuracy}")
        print(f"STDOUT: 학습 완료 — loss={final_loss}, accuracy={final_accuracy}")

        s3 = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )

        if result.checkpoint:
            with result.checkpoint.as_directory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.pt")
                if os.path.exists(model_path):
                    model_key = f"tenants/{tenant_id}/training-jobs/{training_job_id}/model/model.pt"
                    s3.upload_file(model_path, TRAINING_BUCKET, model_key)
                    logging.info(f"[train_job] 모델 저장: {model_key}")
                    print(f"STDOUT: 모델 저장 완료 — {model_key}")

        metrics = {
            "loss": final_loss,
            "accuracy": final_accuracy,
            "epochs": epochs,
        }
        update_status(training_job_id, "COMPLETED", metrics=metrics)
        print(f"RESULT:{json.dumps({'status': 'COMPLETED', 'metrics': metrics})}")

    except Exception as e:
        logging.error(f"[train_job] 학습 실패: {e}")
        print(f"STDOUT: 학습 실패 — {e}")
        update_status(training_job_id, "FAILED", error_msg=str(e))
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()