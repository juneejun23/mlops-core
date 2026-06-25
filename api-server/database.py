from sqlalchemy import create_engine, Column, String, Integer, JSON, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "postgresql://mlops:dashlove@postgres-cluster-rw.mlops-backend.svc.cluster.local:5432/mlops"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Task(Base):
    __tablename__ = "task"

    task_id = Column(String, primary_key=True)
    tenant_id = Column(String, nullable=False)
    model_id = Column(String, nullable=False)
    num_image = Column(Integer, nullable=False)
    status = Column(String, nullable=False, default="PENDING")
    result_url = Column(String, default="")
    meta = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class Outbox(Base):
    __tablename__ = "outbox"

    outbox_id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String, nullable=False)
    model_id = Column(String, nullable=False)
    type = Column(String, nullable=False)
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

Base.metadata.create_all(bind=engine)