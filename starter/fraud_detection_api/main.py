import os
import shutil

import boto3
from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel


class InputData(BaseModel):
    features: list[float]

app = FastAPI()

# Initialize Spark session
spark = SparkSession.builder.appName("FraudDetectionAPI").getOrCreate()

model = None


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith('s3://'):
        raise ValueError('MODEL_S3_URI must start with s3://')

    path = s3_uri[5:]
    bucket, separator, key_prefix = path.partition('/')
    if not bucket or not separator or not key_prefix:
        raise ValueError('MODEL_S3_URI must be in the format s3://bucket/prefix')

    return bucket, key_prefix.rstrip('/')


def download_model_from_s3(s3_uri: str, local_dir: str) -> str:
    bucket, key_prefix = parse_s3_uri(s3_uri)
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')

    downloaded_any_file = False
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get('Contents', []):
            object_key = obj['Key']
            if object_key.endswith('/'):
                continue

            relative_path = object_key[len(key_prefix):].lstrip('/')
            local_file_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            s3_client.download_file(bucket, object_key, local_file_path)
            downloaded_any_file = True

    if not downloaded_any_file:
        raise FileNotFoundError(f'No model artifacts found at {s3_uri}')

    return local_dir

@app.on_event('startup')
async def load_model():
    global model
    model_s3_uri = os.getenv('MODEL_S3_URI')
    local_default_model_path = os.getenv('MODEL_LOCAL_PATH', 'model/fraud_detection_model_latest')
    model_staging_dir = os.getenv('MODEL_STAGING_DIR', '/tmp/fraud_detection_model_latest')

    if model_s3_uri:
        if os.path.exists(model_staging_dir):
            shutil.rmtree(model_staging_dir)
        os.makedirs(model_staging_dir, exist_ok=True)
        model_path = download_model_from_s3(model_s3_uri, model_staging_dir)
    else:
        model_path = local_default_model_path

    # Load the PySpark model
    model = PipelineModel.load(model_path)

@app.post('/predict/')
def predict(data: InputData):
    # Declaring Feature Columns
    columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

    # Create a Spark DataFrame for prediction
    df = spark.createDataFrame([data.features], columns)

    # Make prediction
    predictions = model.transform(df)

    prediction = predictions.select("prediction").collect()[0][0]
    
    return {'prediction': prediction}
