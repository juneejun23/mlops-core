import os
import io
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import boto3
from botocore.client import Config
import ray
import ray.train
from ray import train
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
from ray.data import ActorPoolStrategy

MINIO_ENDPOINT = "http://minio-api.100.112.196.74.nip.io:31135"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "quHCnPBfDaYU0UsV0vfM"
TRAINING_BUCKET = "training-data"
JOB_ID = "training-2265da1c1901"
TENANT_ID = "user01"


# =============================================
# Ray Data Actor — MinIO에서 .jpg 읽어서 전처리
# =============================================

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

            # xception 전처리 (xception-consumer와 동일)
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


# =============================================
# 모델 정의
# =============================================

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =============================================
# train_loop_per_worker
# =============================================

def train_loop_per_worker(config: dict):
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lr = config["lr"]

    train_shard = train.get_dataset_shard("train")
    model = SimpleClassifier()
    model = ray.train.torch.prepare_model(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

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

        avg_loss = total_loss / max(num_batches, 1)
        ray.train.report({"epoch": epoch, "loss": avg_loss})
        print(f"Epoch {epoch}: loss={avg_loss:.4f}")


# =============================================
# MinIO에서 이미지 경로 목록 가져오기
# =============================================

def get_image_keys():
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    real_keys = []
    fake_keys = []

    for label in ["real", "fake"]:
        prefix = f"tenants/{TENANT_ID}/training-jobs/{JOB_ID}/cropped/{label}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=TRAINING_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                if label == "real":
                    real_keys.append(obj["Key"])
                else:
                    fake_keys.append(obj["Key"])

    print(f"real: {len(real_keys)}, fake: {len(fake_keys)}")
    return real_keys, fake_keys


# =============================================
# 메인
# =============================================

if __name__ == "__main__":
    ray.init(
        address="ray://localhost:10001",
        runtime_env={
            "pip": ["torch==2.2.2", "torchvision==0.17.2", "numpy==1.26.0", "boto3==1.43.45", "Pillow"],
            "env_vars": {
                "AWS_ACCESS_KEY_ID": "minio",
                "AWS_SECRET_ACCESS_KEY": "quHCnPBfDaYU0UsV0vfM",
                "AWS_ENDPOINT_URL": "http://minio-api.100.112.196.74.nip.io:31135",
                "AWS_DEFAULT_REGION": "us-east-1",
            }
        }
    )

    # MinIO에서 이미지 경로 가져오기
    real_keys, fake_keys = get_image_keys()

    items = (
        [{"minio_path": k, "label": "real"} for k in real_keys] +
        [{"minio_path": k, "label": "fake"} for k in fake_keys]
    )

    # Ray Data Dataset 구성
    ds = ray.data.from_items(items)
    ds = ds.map_batches(
        ImagePreprocessor,
        compute=ActorPoolStrategy(size=1),
        batch_size=16,
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={"epochs": 2, "batch_size": 16, "lr": 0.001},
        datasets={"train": ds},
        scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
        run_config=RunConfig(
            storage_path="s3://training-data/ray-checkpoints",
        ),
    )

    result = trainer.fit()
    print(f"완료: {result.metrics}")
    ray.shutdown()
