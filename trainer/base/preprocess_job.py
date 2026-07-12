import os
import sys
import json
import zipfile
import logging
import shutil
import io
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import boto3
from botocore.client import Config
import ray
from ray.data import ActorPoolStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://minio.minio.svc.cluster.local:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "quHCnPBfDaYU0UsV0vfM")
TRAINING_BUCKET = os.environ.get("TRAINING_BUCKET", "training-data")
SKIP_THRESHOLD = float(os.environ.get("SKIP_THRESHOLD", "0.5"))


class MTCNNCropper:
    def __init__(self):
        from facenet_pytorch import MTCNN as FacenetMTCNN
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = FacenetMTCNN(keep_all=True, device=device)
        self.s3 = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
        logging.info(f"MTCNNCropper initialized on {device}")

    def __call__(self, batch: dict) -> dict:
        keys = []
        labels_out = []

        for i in range(len(batch["minio_path"])):
            minio_path = batch["minio_path"][i]  # MinIO 경로
            label = batch["label"][i]
            filename = batch["filename"][i]
            training_job_id = batch["training_job_id"][i]
            tenant_id = batch["tenant_id"][i]
            face_based = batch["face_based"][i]

            try:
                # MinIO에서 직접 이미지 읽기
                response = self.s3.get_object(Bucket=TRAINING_BUCKET, Key=minio_path)
                img_bytes = response["Body"].read()
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img_array = np.array(img)

                if face_based:
                    boxes, _ = self.detector.detect(img_array)
                    if boxes is None or len(boxes) == 0:
                        keys.append(None)
                        labels_out.append(label)
                        continue
                    box = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
                    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                    cropped = img_array[y1:y2, x1:x2]
                    if cropped.size == 0:
                        keys.append(None)
                        labels_out.append(label)
                        continue
                    img = Image.fromarray(cropped)

                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                object_key = f"tenants/{tenant_id}/training-jobs/{training_job_id}/cropped/{label}/{filename}"
                self.s3.put_object(
                    Bucket=TRAINING_BUCKET,
                    Key=object_key,
                    Body=buf.getvalue(),
                )
                keys.append(object_key)
                labels_out.append(label)

            except Exception as e:
                logging.error(f"Error processing {filename}: {e}")
                keys.append(None)
                labels_out.append(label)

        return {"key": keys, "label": labels_out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=str, required=True)
    args = parser.parse_args()

    payload = json.loads(args.payload)
    training_job_id = payload["training_job_id"]
    zip_path = payload["zip_path"]
    tenant_id = payload["tenant_id"]
    face_based = payload.get("face_based", False)

    ray.init()

    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    work_dir = f"/tmp/{training_job_id}"
    zip_local = f"{work_dir}/upload.zip"
    extract_dir = f"{work_dir}/data"

    try:
        # 1. zip 다운로드
        logging.info(f"[1/4] Downloading zip: {zip_path}")
        os.makedirs(work_dir, exist_ok=True)
        s3.download_file(TRAINING_BUCKET, zip_path, zip_local)

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

        # 4. Ray Data로 병렬 크롭 + MinIO 업로드
        logging.info(f"[4/4] Cropping with Ray Data (face_based={face_based})")

        all_images = (
            list(Path(extract_dir).rglob("*.jpg")) +
            list(Path(extract_dir).rglob("*.png")) +
            list(Path(extract_dir).rglob("*.jpeg"))
        )

        items = [
            {
                "path": str(img_path),
                "label": labels[img_path.name],
                "filename": img_path.name,
                "training_job_id": training_job_id,
                "tenant_id": tenant_id,
                "face_based": face_based,
            }
            for img_path in all_images
            if img_path.name in labels
        ]

        total = len(items)

        ds = ray.data.from_items(items)
        result_ds = ds.map_batches(
            MTCNNCropper,
            compute=ActorPoolStrategy(size=1),
            batch_size=16,
        )

        results = result_ds.take_all()
        real_keys = [r["key"] for r in results if r["label"] == "real" and r["key"] is not None]
        fake_keys = [r["key"] for r in results if r["label"] == "fake" and r["key"] is not None]
        skipped = len([r for r in results if r["key"] is None])

        if total > 0:
            skip_ratio = skipped / total
            logging.info(f"Skip ratio: {skip_ratio:.1%} ({skipped}/{total})")
            if skip_ratio >= SKIP_THRESHOLD:
                raise Exception(f"Too many skipped: {skip_ratio:.1%} >= {SKIP_THRESHOLD:.1%}")

        logging.info(f"Cropped: real={len(real_keys)}, fake={len(fake_keys)}, skipped={skipped}")

        # 결과를 stdout으로 출력 — consumer가 읽어서 train-topic 발행
        result = {
            "real_keys": real_keys,
            "fake_keys": fake_keys,
            "skipped": skipped,
        }
        print(f"RESULT:{json.dumps(result)}")

    finally:
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            logging.info(f"Cleaned up: {work_dir}")
        ray.shutdown()


if __name__ == "__main__":
    main()