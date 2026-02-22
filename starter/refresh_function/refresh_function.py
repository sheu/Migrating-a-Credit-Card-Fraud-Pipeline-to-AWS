import os
from datetime import datetime, timezone
from urllib.parse import unquote_plus

import boto3


glue_client = boto3.client('glue')
s3_client = boto3.client('s3')


def build_archive_key(source_key: str, archive_prefix: str) -> str:
    filename = source_key.split('/')[-1]
    name, _ = os.path.splitext(filename)
    date_suffix = datetime.now(timezone.utc).strftime('%Y%m%d')
    return f"{archive_prefix.rstrip('/')}/{name}_retrain_{date_suffix}.csv"


def process_s3_record(record: dict) -> dict:
    bucket_name = record['s3']['bucket']['name']
    object_key = unquote_plus(record['s3']['object']['key'])

    if not object_key.endswith('.csv') or 'retrain' in object_key:
        return {'status': 'skipped', 'key': object_key}

    glue_job_name = os.environ['GLUE_JOB_NAME']
    model_save_path = os.environ['MODEL_SAVE_PATH']
    archive_prefix = os.getenv('ARCHIVE_PREFIX', 'archive')

    data_url = f's3://{bucket_name}/{object_key}'

    response = glue_client.start_job_run(
        JobName=glue_job_name,
        Arguments={
            '--DATA_URL': data_url,
            '--MODEL_SAVE_PATH': model_save_path,
            '--JOB_NAME': glue_job_name,
        },
    )

    archive_key = build_archive_key(object_key, archive_prefix)
    s3_client.copy_object(
        Bucket=bucket_name,
        CopySource={'Bucket': bucket_name, 'Key': object_key},
        Key=archive_key,
    )
    s3_client.delete_object(Bucket=bucket_name, Key=object_key)

    return {
        'status': 'processed',
        'key': object_key,
        'archive_key': archive_key,
        'glue_job_run_id': response['JobRunId'],
    }


def lambda_handler(event, context):
    results = []
    for record in event.get('Records', []):
        if record.get('eventSource') != 'aws:s3':
            continue
        results.append(process_s3_record(record))

    return {'results': results}

