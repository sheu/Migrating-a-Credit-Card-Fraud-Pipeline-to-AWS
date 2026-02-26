import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel


class InputData(BaseModel):
    features: list[float]


app = FastAPI()

spark = None

model = None


def _create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("FraudDetectionAPI")
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .getOrCreate()
    )


def _get_spark_session() -> SparkSession:
    global spark
    if spark is None:
        spark = _create_spark_session()
    return spark


def _reset_spark_session() -> SparkSession:
    global spark
    try:
        if spark is not None:
            spark.stop()
    except Exception:
        pass
    spark = _create_spark_session()
    return spark


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    if p.scheme != "s3" or not p.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return p.netloc, p.path.lstrip("/")


def _download_s3_prefix(s3_uri: str, dst_dir: Path) -> Path:
    import boto3

    bucket, prefix = _parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")

    local_root = dst_dir / Path(prefix).name
    local_root.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    found = False

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            found = True
            key = obj["Key"]
            rel = key[len(prefix):].lstrip("/")
            if not rel:
                continue
            out = local_root / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(out))

    if not found:
        raise FileNotFoundError(f"No objects found under {s3_uri}")

    return local_root


def _load_model():
    global model

    model_s3_uri = os.getenv(
        "MODEL_S3_URI",
        "s3://fraud-pipeline-sg-data-bucket/models/fraud_detection_model_latest/",
    )
    model_path = os.getenv("MODEL_PATH", "model/fraud_detection_model_latest")

    if model_s3_uri:
        print(f"Model load source: s3 ({model_s3_uri})")
        tmp = Path(tempfile.mkdtemp(prefix="fraud-model-"))
        local_dir = _download_s3_prefix(model_s3_uri.rstrip("/"), tmp)
        print(f"Model load resolved local path: {local_dir}")
        model = PipelineModel.load(str(local_dir))
        print("Model load succeeded from S3 source")
    else:
        print(f"Model load source: local ({model_path})")
        model = PipelineModel.load(model_path)
        print("Model load succeeded from local source")


@app.on_event("startup")
async def startup():
    try:
        _load_model()
    except Exception as e:
        # Let the container start; endpoint will 500 with a readable error
        print(f"Model load failed at startup: {e}")


@app.post("/predict/")
def predict(data: InputData):
    global model

    if model is None:
        try:
            _load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    columns = [
        "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
        "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
        "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
        "Amount",
    ]

    if len(data.features) != len(columns):
        raise HTTPException(status_code=400, detail=f"Expected {len(columns)} features, got {len(data.features)}")

    spark_session = _get_spark_session()
    try:
        df = spark_session.createDataFrame([data.features], columns)
        pred = model.transform(df).select("prediction").collect()[0][0]
    except Exception as e:
        # Recover once from common Spark gateway crashes/connection resets.
        err_text = str(e)
        recoverable_markers = (
            "Connection refused",
            "Answer from Java side is empty",
            "Connection reset by peer",
            "Py4JNetworkError",
            "EOFError",
        )
        is_recoverable = isinstance(e, EOFError) or any(marker in err_text for marker in recoverable_markers)

        if not is_recoverable:
            raise HTTPException(status_code=500, detail=str(e))

        try:
            print(f"Recoverable Spark error encountered, resetting Spark session: {e}")
            spark_session = _reset_spark_session()
            df = spark_session.createDataFrame([data.features], columns)
            pred = model.transform(df).select("prediction").collect()[0][0]
        except Exception as retry_error:
            raise HTTPException(status_code=500, detail=str(retry_error))

    return {"prediction": float(pred)}


@app.get("/health")
def health():
    return {"status": "ok"}