from minio import Minio
from datetime import timedelta

MINIO_ENDPOINT = "minio-api.100.112.196.74.nip.io:31135"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "quHCnPBfDaYU0UsV0vfM"
BUCKET_NAME = "hyjk826-mlops-1011"

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)


def ensure_bucket():
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)


def get_upload_url(object_key: str) -> str:
    return minio_client.presigned_put_object(
        BUCKET_NAME,
        object_key,
        expires=timedelta(minutes=5),
    )


ensure_bucket()