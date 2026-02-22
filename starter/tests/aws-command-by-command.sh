# Resource creation file.

export AWS_REGION=us-east-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

export PROJECT=fraud-mvp-dev
export DATA_BUCKET=${PROJECT}-data-${ACCOUNT_ID}
export MODEL_BUCKET=${PROJECT}-model-${ACCOUNT_ID}

export ECR_REPO=fraud-detection-api
export ECS_CLUSTER=${PROJECT}-cluster
export ECS_SERVICE=${PROJECT}-service
export TASK_FAMILY=${PROJECT}-task
export ALB_NAME=${PROJECT}-alb
export TG_NAME=${PROJECT}-tg

export GLUE_JOB_NAME=${PROJECT}-glue-train
export LAMBDA_NAME=${PROJECT}-retrain-trigger

export IMAGE_TAG=latest

### 1) Create S3 buckets and upload project assets

aws s3 mb s3://${DATA_BUCKET} --region ${AWS_REGION}
aws s3 mb s3://${MODEL_BUCKET} --region ${AWS_REGION}

aws s3api put-object --bucket ${DATA_BUCKET} --key raw/
aws s3api put-object --bucket ${DATA_BUCKET} --key retrain/
aws s3api put-object --bucket ${DATA_BUCKET} --key scripts/
aws s3api put-object --bucket ${MODEL_BUCKET} --key model/latest/

aws s3 cp starter/data/creditcard.csv s3://${DATA_BUCKET}/raw/creditcard.csv
aws s3 cp starter/fraud_detection_pipeline/fraud_detector_model_trainer.py s3://${DATA_BUCKET}/scripts/fraud_detector_model_trainer.py

### 2) Create IAM roles (Glue, Lambda, ECS)

cat > trust-glue.json << 'EOF'
{
  "Version":"2012-10-17",
  "Statement":[{"Effect":"Allow","Principal":{"Service":"glue.amazonaws.com"},"Action":"sts:AssumeRole"}]
}
EOF

cat > trust-lambda.json << 'EOF'
{
  "Version":"2012-10-17",
  "Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]
}
EOF

cat > trust-ecs-task.json << 'EOF'
{
  "Version":"2012-10-17",
  "Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]
}
EOF

aws iam create-role --role-name ${PROJECT}-glue-role --assume-role-policy-document file://trust-glue.json
aws iam create-role --role-name ${PROJECT}-lambda-role --assume-role-policy-document file://trust-lambda.json
aws iam create-role --role-name ${PROJECT}-ecs-exec-role --assume-role-policy-document file://trust-ecs-task.json
aws iam create-role --role-name ${PROJECT}-ecs-task-role --assume-role-policy-document file://trust-ecs-task.json

aws iam attach-role-policy --role-name ${PROJECT}-glue-role --policy-arn arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole
aws iam attach-role-policy --role-name ${PROJECT}-lambda-role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam attach-role-policy --role-name ${PROJECT}-ecs-exec-role --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

cat > project-s3-policy.json << EOF
{
  "Version":"2012-10-17",
  "Statement":[
    {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:ListBucket"],"Resource":[
      "arn:aws:s3:::${DATA_BUCKET}",
      "arn:aws:s3:::${DATA_BUCKET}/*",
      "arn:aws:s3:::${MODEL_BUCKET}",
      "arn:aws:s3:::${MODEL_BUCKET}/*"
    ]}
  ]
}
EOF

aws iam put-role-policy --role-name ${PROJECT}-glue-role --policy-name ${PROJECT}-glue-s3 --policy-document file://project-s3-policy.json
aws iam put-role-policy --role-name ${PROJECT}-ecs-task-role --policy-name ${PROJECT}-ecs-s3 --policy-document file://project-s3-policy.json

cat > lambda-glue-policy.json << EOF
{
  "Version":"2012-10-17",
  "Statement":[
    {"Effect":"Allow","Action":["glue:StartJobRun"],"Resource":"*"},
    {"Effect":"Allow","Action":["s3:GetObject","s3:ListBucket"],"Resource":[
      "arn:aws:s3:::${DATA_BUCKET}",
      "arn:aws:s3:::${DATA_BUCKET}/*"
    ]}
  ]
}
EOF

aws iam put-role-policy --role-name ${PROJECT}-lambda-role --policy-name ${PROJECT}-lambda-glue --policy-document file://lambda-glue-policy.json

### 3) Build/push API image to ECR
aws ecr create-repository --repository-name ${ECR_REPO} --region ${AWS_REGION} || true

aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

docker build -t ${ECR_REPO}:${IMAGE_TAG} starter/fraud_detection_api
docker tag ${ECR_REPO}:${IMAGE_TAG} ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}
docker push ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}

### 4) Deploy ECS Fargate + ALB
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SUBNET_1=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=${VPC_ID} --query 'Subnets[0].SubnetId' --output text)
SUBNET_2=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=${VPC_ID} --query 'Subnets[1].SubnetId' --output text)
SG_ID=$(aws ec2 describe-security-groups --filters Name=vpc-id,Values=${VPC_ID} Name=group-name,Values=default --query 'SecurityGroups[0].GroupId' --output text)

aws ecs create-cluster --cluster-name ${ECS_CLUSTER}

cat > taskdef.json << EOF
{
  "family": "${TASK_FAMILY}",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::${ACCOUNT_ID}:role/${PROJECT}-ecs-exec-role",
  "taskRoleArn": "arn:aws:iam::${ACCOUNT_ID}:role/${PROJECT}-ecs-task-role",
  "containerDefinitions": [
    {
      "name": "fraud-api",
      "image": "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}",
      "essential": true,
      "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
      "environment": [
        {"name":"MODEL_S3_PATH","value":"s3://${MODEL_BUCKET}/model/latest/"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-region": "${AWS_REGION}",
          "awslogs-group": "/ecs/${PROJECT}",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
EOF

aws logs create-log-group --log-group-name /ecs/${PROJECT} || true
aws ecs register-task-definition --cli-input-json file://taskdef.json

aws elbv2 create-load-balancer \
  --name ${ALB_NAME} \
  --subnets ${SUBNET_1} ${SUBNET_2} \
  --security-groups ${SG_ID} \
  --scheme internet-facing \
  --type application

ALB_ARN=$(aws elbv2 describe-load-balancers --names ${ALB_NAME} --query 'LoadBalancers[0].LoadBalancerArn' --output text)

aws elbv2 create-target-group \
  --name ${TG_NAME} \
  --protocol HTTP \
  --port 8000 \
  --target-type ip \
  --vpc-id ${VPC_ID}

TG_ARN=$(aws elbv2 describe-target-groups --names ${TG_NAME} --query 'TargetGroups[0].TargetGroupArn' --output text)

aws elbv2 create-listener \
  --load-balancer-arn ${ALB_ARN} \
  --protocol HTTP \
  --port 80 \
  --default-actions Type=forward,TargetGroupArn=${TG_ARN}

  aws ecs create-service \
  --cluster ${ECS_CLUSTER} \
  --service-name ${ECS_SERVICE} \
  --task-definition ${TASK_FAMILY} \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[${SUBNET_1},${SUBNET_2}],securityGroups=[${SG_ID}],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=${TG_ARN},containerName=fraud-api,containerPort=8000"

### 5) Create Glue training job

  aws glue create-job \
  --name ${GLUE_JOB_NAME} \
  --role arn:aws:iam::${ACCOUNT_ID}:role/${PROJECT}-glue-role \
  --command Name=glueetl,ScriptLocation=s3://${DATA_BUCKET}/scripts/fraud_detector_model_trainer.py,PythonVersion=3 \
  --glue-version 4.0 \
  --worker-type G.1X \
  --number-of-workers 2 \
  --default-arguments "{\"--DATA_URL\":\"s3://${DATA_BUCKET}/raw/creditcard.csv\",\"--MODEL_SAVE_PATH\":\"s3://${MODEL_BUCKET}/model/latest/\",\"--JOB_NAME\":\"${GLUE_JOB_NAME}\"}"

  aws glue start-job-run --job-name ${GLUE_JOB_NAME}

  ### 6) Create Lambda trigger for retrain files

cat > lambda_function.py << 'EOF'
import os
import boto3
glue = boto3.client("glue")
JOB = os.environ["GLUE_JOB_NAME"]

def lambda_handler(event, context):
    glue.start_job_run(JobName=JOB)
    return {"statusCode": 200, "body": "Glue job started"}
EOF

zip lambda.zip lambda_function.py

aws lambda create-function \
  --function-name ${LAMBDA_NAME} \
  --runtime python3.12 \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://lambda.zip \
  --role arn:aws:iam::${ACCOUNT_ID}:role/${PROJECT}-lambda-role \
  --environment "Variables={GLUE_JOB_NAME=${GLUE_JOB_NAME}}"

LAMBDA_ARN=$(aws lambda get-function --function-name ${LAMBDA_NAME} --query 'Configuration.FunctionArn' --output text)

aws lambda add-permission \
  --function-name ${LAMBDA_NAME} \
  --statement-id s3invoke \
  --action lambda:InvokeFunction \
  --principal s3.amazonaws.com \
  --source-arn arn:aws:s3:::${DATA_BUCKET}

aws s3api put-bucket-notification-configuration \
  --bucket ${DATA_BUCKET} \
  --notification-configuration "{
    \"LambdaFunctionConfigurations\": [
      {
        \"LambdaFunctionArn\": \"${LAMBDA_ARN}\",
        \"Events\": [\"s3:ObjectCreated:*\"] ,
        \"Filter\": {\"Key\": {\"FilterRules\": [{\"Name\":\"prefix\",\"Value\":\"retrain/\"}]}}
      }
    ]
  }"

  ### 7) Validate end-to-end
  aws s3 cp starter/data/creditcard.csv s3://${DATA_BUCKET}/retrain/retrain_$(date +%s).csv

  ALB_DNS=$(aws elbv2 describe-load-balancers --names ${ALB_NAME} --query 'LoadBalancers[0].DNSName' --output text)
curl -X POST "http://${ALB_DNS}/predict/" -H "Content-Type: application/json" -d '{"features":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}'