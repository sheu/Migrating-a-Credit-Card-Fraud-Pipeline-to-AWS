# Creating and loading AWS Lambda function
1. Create a Lambda function in the AWS Management Console.
2. Upload the Lambda function code.
3. Set up an S3 event trigger for the desired bucket and prefix

## Minimum IAM policy for Lambda
Use `lambda_iam_policy.json` in this folder and replace placeholders:
- `<AWS_REGION>`
- `<ACCOUNT_ID>`
- `<GLUE_JOB_NAME>`
- `<DATA_BUCKET>`
- `<RETRAIN_PREFIX>` (for example: `retrain`)
- `<ARCHIVE_PREFIX>` (for example: `archive`)

Attach policy to your Lambda role:

```bash
aws iam put-role-policy \
	--role-name <LAMBDA_ROLE_NAME> \
	--policy-name refresh-function-inline-policy \
	--policy-document file://starter/refresh_function/lambda_iam_policy.json
```

## Required Lambda environment variables
- `GLUE_JOB_NAME`
- `MODEL_SAVE_PATH` (example: `s3://<model-bucket>/model/latest/`)
- `ARCHIVE_PREFIX` (optional, default: `archive`)