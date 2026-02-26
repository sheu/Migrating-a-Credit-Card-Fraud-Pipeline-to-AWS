import os
import boto3

glue = boto3.client("glue")


def handler(event, context):
    """
    Triggered by S3:ObjectCreated events on the raw data prefix.
    Starts the Glue job and passes S3 locations as job arguments.
    """
    job_name = os.environ["GLUE_JOB_NAME"]
    input_s3 = os.environ["INPUT_S3"]           # e.g. s3://bucket/raw/
    processed_s3 = os.environ["PROCESSED_S3"]   # e.g. s3://bucket/processed/
    model_s3 = os.environ["MODEL_S3"]           # e.g. s3://bucket/models/fraud_detection_model_latest

    # Optional: capture the specific object that triggered retraining (useful for auditing)
    record = event["Records"][0]
    bucket = record["s3"]["bucket"]["name"]
    key = record["s3"]["object"]["key"]
    trigger_object = f"s3://{bucket}/{key}"

    resp = glue.start_job_run(
        JobName=job_name,
        Arguments={
            "--INPUT_S3": input_s3,
            "--PROCESSED_S3": processed_s3,
            "--MODEL_S3": model_s3,
            "--TRIGGER_OBJECT": trigger_object,  # unused by job unless you add it
        },
    )

    return {"status": "started", "jobRunId": resp["JobRunId"], "trigger": trigger_object}