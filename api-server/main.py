from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db, Task, Outbox
from storage import get_upload_url, get_training_upload_url
import uuid
from database import get_db, Task, Outbox, TrainingJob
from datetime import datetime, timedelta

app = FastAPI()


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

    # 멱등성 보장 — outbox에 이미 같은 이벤트가 있으면 중복 삽입 안 함
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
