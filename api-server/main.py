from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db, Task, Outbox
from storage import get_upload_url, get_training_upload_url
import uuid
import os
import json
import boto3
from botocore.client import Config
from database import get_db, Task, Outbox, TrainingJob
from datetime import datetime, timedelta
from ray.job_submission import JobSubmissionClient

app = FastAPI()

RAY_DASHBOARD_URL = os.environ.get(
    "RAY_DASHBOARD_URL",
    "http://ray-cluster-kuberay-head-svc.kuberay.svc.cluster.local:8265"
)


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/presigned-url")
def presigned_url(tenant_id: str, task_id: str, filename: str):
    object_key = f"tenants/{tenant_id}/tasks/{task_id}/input/{filename}"
    upload_url = get_upload_url(object_key)
    return {"upload_url": upload_url, "object_key": object_key}

@app.post("/tasks")
def create_task(tenant_id: str, model_id: str, num_image: int, db: Session = Depends(get_db)):
    task_id = f"task-{uuid.uuid4().hex[:12]}"
    new_task = Task(
        task_id=task_id,
        tenant_id=tenant_id,
        model_id=model_id,
        num_image=num_image,
    )
    db.add(new_task)
    db.commit()
    return {"task_id": task_id, "status": "PENDING"}

@app.post("/tasks/{task_id}/complete")
def upload_complete(task_id: str, object_key: str, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.task_id == task_id).first()
    if task is None:
        return {"error": "task not found"}

    task.status = "QUEUED"

    new_event = Outbox(
        task_id=task.task_id,
        model_id=task.model_id,
        type="inference-request",
        payload={
            "task_id": task.task_id,
            "input_key": object_key,
            "num_image": task.num_image,
        },
    )
    db.add(new_event)
    db.commit()

    return {"task_id": task_id, "status": "QUEUED"}

@app.get("/tasks/{task_id}")
def get_task(task_id: str, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.task_id == task_id).first()
    if task is None:
        return {"error": "task not found"}
    return {
        "task_id": task.task_id,
        "tenant_id": task.tenant_id,
        "model_id": task.model_id,
        "status": task.status,
        "result_url": task.result_url,
    }

@app.post("/training-jobs")
def create_training_job(
    tenant_id: str,
    architecture: str,
    epochs: int,
    batch_size: int,
    lr: float,
    face_based: bool,
    db: Session = Depends(get_db)
):
    training_job_id = f"training-{uuid.uuid4().hex[:12]}"
    new_job = TrainingJob(
        training_job_id=training_job_id,
        tenant_id=tenant_id,
        architecture=architecture,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        face_based=face_based,
    )
    db.add(new_job)
    db.commit()
    return {"training_job_id": training_job_id, "status": "PENDING"}


@app.post("/training-jobs/{training_job_id}/presigned-url")
def get_training_presigned_url(
    training_job_id: str,
    tenant_id: str,
    db: Session = Depends(get_db)
):
    job = db.query(TrainingJob).filter(
        TrainingJob.training_job_id == training_job_id
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    object_key = f"tenants/{tenant_id}/training-jobs/{training_job_id}/input/upload.zip"
    upload_url = get_training_upload_url(object_key)

    job.zip_path = object_key
    job.updated_at = datetime.utcnow()
    db.commit()

    return {"upload_url": upload_url, "object_key": object_key}


@app.get("/training-jobs/{training_job_id}")
def get_training_job(training_job_id: str, db: Session = Depends(get_db)):
    job = db.query(TrainingJob).filter(
        TrainingJob.training_job_id == training_job_id
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job

@app.post("/training-jobs/{training_job_id}/complete")
def complete_training_job(training_job_id: str, db: Session = Depends(get_db)):
    job = db.query(TrainingJob).filter(
        TrainingJob.training_job_id == training_job_id
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    existing = db.query(Outbox).filter(
        Outbox.task_id == training_job_id,
        Outbox.type == "preprocess-request"
    ).first()
    if existing:
        return {"training_job_id": training_job_id, "status": job.status, "message": "Already queued"}

    job.status = "QUEUED"
    job.updated_at = datetime.utcnow()

    new_event = Outbox(
        task_id=training_job_id,
        model_id=job.architecture,
        type="preprocess-request",
        payload={
            "training_job_id": training_job_id,
            "zip_path": job.zip_path,
            "architecture": job.architecture,
            "epochs": job.epochs,
            "batch_size": job.batch_size,
            "lr": job.lr,
            "face_based": job.face_based,
        },
    )
    db.add(new_event)
    db.commit()

    return {"training_job_id": training_job_id, "status": "QUEUED"}


@app.post("/training-jobs/{training_job_id}/train")
def start_training(training_job_id: str, db: Session = Depends(get_db)):
    job = db.query(TrainingJob).filter(
        TrainingJob.training_job_id == training_job_id
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    # MinIO에서 cropped 이미지 경로 목록 가져오기
    s3 = boto3.client(
        "s3",
        endpoint_url="http://minio.minio.svc.cluster.local:9000",
        aws_access_key_id="minio",
        aws_secret_access_key="quHCnPBfDaYU0UsV0vfM",
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    real_keys = []
    fake_keys = []
    for label in ["real", "fake"]:
        prefix = f"tenants/{job.tenant_id}/training-jobs/{training_job_id}/cropped/{label}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket="training-data", Prefix=prefix):
            for obj in page.get("Contents", []):
                if label == "real":
                    real_keys.append(obj["Key"])
                else:
                    fake_keys.append(obj["Key"])

    if not real_keys and not fake_keys:
        raise HTTPException(status_code=400, detail="No cropped images found. Run preprocessing first.")

    # Ray Job 제출
    payload = {
        "training_job_id": training_job_id,
        "real_keys": real_keys,
        "fake_keys": fake_keys,
        "epochs": job.epochs,
        "batch_size": job.batch_size,
        "lr": job.lr,
        "tenant_id": job.tenant_id,
    }

    ray_client = JobSubmissionClient(RAY_DASHBOARD_URL)
    ray_job_id = ray_client.submit_job(
        entrypoint=f"python train_job.py --payload '{json.dumps(payload)}'",
        runtime_env={
            "working_dir": "/app",
            "pip": ["torch==2.2.2", "torchvision==0.17.2", "numpy==1.26.0",
                    "boto3==1.43.45", "Pillow", "sqlalchemy", "psycopg2-binary"],
        }
    )

    # DB에 ray_job_id 저장
    job.ray_job_id = ray_job_id
    job.status = "TRAINING"
    job.updated_at = datetime.utcnow()
    db.commit()

    return {
        "training_job_id": training_job_id,
        "ray_job_id": ray_job_id,
        "status": "TRAINING"
    }


@app.get("/training-jobs/{training_job_id}/status")
def get_training_status(training_job_id: str, db: Session = Depends(get_db)):
    job = db.query(TrainingJob).filter(
        TrainingJob.training_job_id == training_job_id
    ).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    if job.ray_job_id:
        ray_client = JobSubmissionClient(RAY_DASHBOARD_URL)
        ray_status = ray_client.get_job_status(job.ray_job_id)
        return {
            "training_job_id": training_job_id,
            "status": job.status,
            "ray_job_id": job.ray_job_id,
            "ray_status": str(ray_status),
            "metrics": job.metrics,
        }

    return {
        "training_job_id": training_job_id,
        "status": job.status,
        "metrics": job.metrics,
    }