from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from database import get_db, Task
from storage import get_upload_url
import uuid

app = FastAPI()


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/presigned-url")
def presigned_url(task_id: str, filename: str):
    object_key = f"tenants/dummy/tasks/{task_id}/input/{filename}"
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