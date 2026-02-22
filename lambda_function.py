import os
import boto3
glue = boto3.client("glue")
JOB = os.environ["GLUE_JOB_NAME"]

def lambda_handler(event, context):
    glue.start_job_run(JobName=JOB)
    return {"statusCode": 200, "body": "Glue job started"}
