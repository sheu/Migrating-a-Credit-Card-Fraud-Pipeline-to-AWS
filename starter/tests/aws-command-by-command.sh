# Resource creation file.
set -euo pipefail

log() {
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $1"
}

wait_for_ecs_service_stable() {
  local cluster="$1"
  local service="$2"
  local max_checks="${3:-30}"
  local sleep_seconds="${4:-20}"

  log "Waiting for ECS service to stabilize (max $((max_checks * sleep_seconds))s)..."

  local check
  for ((check = 1; check <= max_checks; check++)); do
    local status running desired pending deployments

    status=$(aws ecs describe-services --cluster "${cluster}" --services "${service}" --query 'services[0].status' --output text 2>/dev/null || echo "UNKNOWN")
    running=$(aws ecs describe-services --cluster "${cluster}" --services "${service}" --query 'services[0].runningCount' --output text 2>/dev/null || echo "0")
    desired=$(aws ecs describe-services --cluster "${cluster}" --services "${service}" --query 'services[0].desiredCount' --output text 2>/dev/null || echo "0")
    pending=$(aws ecs describe-services --cluster "${cluster}" --services "${service}" --query 'services[0].pendingCount' --output text 2>/dev/null || echo "0")
    deployments=$(aws ecs describe-services --cluster "${cluster}" --services "${service}" --query 'length(services[0].deployments)' --output text 2>/dev/null || echo "99")

    log "ECS check ${check}/${max_checks}: status=${status}, running=${running}, desired=${desired}, pending=${pending}, deployments=${deployments}"

    if [ "${status}" = "ACTIVE" ] && [ "${running}" = "${desired}" ] && [ "${pending}" = "0" ] && [ "${deployments}" = "1" ]; then
      log "ECS service is stable."
      return 0
    fi

    sleep "${sleep_seconds}"
  done

  log "ECS service did not stabilize in time. Printing diagnostics..."
  aws ecs describe-services --cluster "${cluster}" --services "${service}" --query 'services[0].events[0:10].[createdAt,message]' --output table || true

  local stopped_task_arn
  stopped_task_arn=$(aws ecs list-tasks --cluster "${cluster}" --service-name "${service}" --desired-status STOPPED --max-results 1 --query 'taskArns[0]' --output text 2>/dev/null || true)
  if [ -n "${stopped_task_arn}" ] && [ "${stopped_task_arn}" != "None" ]; then
    aws ecs describe-tasks --cluster "${cluster}" --tasks "${stopped_task_arn}" --query 'tasks[0].{StoppedReason:stoppedReason,Containers:containers[*].{Name:name,Reason:reason,ExitCode:exitCode}}' --output json || true
  fi

  if [ -n "${TG_ARN:-}" ] && [ "${TG_ARN}" != "None" ]; then
    aws elbv2 describe-target-health --target-group-arn "${TG_ARN}" --output table || true
  fi

  return 1
}

wait_for_lambda_update_ready() {
  local function_name="$1"
  local max_checks="${2:-30}"
  local sleep_seconds="${3:-5}"

  local check
  for ((check = 1; check <= max_checks; check++)); do
    local status
    status=$(aws lambda get-function-configuration --function-name "${function_name}" --query 'LastUpdateStatus' --output text 2>/dev/null || echo "Unknown")

    if [ "${status}" = "Successful" ] || [ "${status}" = "Unknown" ]; then
      return 0
    fi

    log "Lambda ${function_name} update status=${status}; waiting (${check}/${max_checks})"
    sleep "${sleep_seconds}"
  done

  log "Lambda ${function_name} is still updating after timeout."
  return 1
}

safe_lambda_update_configuration() {
  local function_name="$1"
  shift

  wait_for_lambda_update_ready "${function_name}" 30 5
  aws lambda update-function-configuration --function-name "${function_name}" "$@"
  wait_for_lambda_update_ready "${function_name}" 30 5
}

trap 'log "ERROR: Script failed near line ${LINENO}."' ERR

log "Starting AWS MVP deployment runbook..."

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
log "STEP 1/7: Creating S3 buckets and uploading project assets"

aws s3 mb s3://${DATA_BUCKET} --region ${AWS_REGION} || true
aws s3 mb s3://${MODEL_BUCKET} --region ${AWS_REGION} || true

aws s3api put-object --bucket ${DATA_BUCKET} --key raw/
aws s3api put-object --bucket ${DATA_BUCKET} --key retrain/
aws s3api put-object --bucket ${DATA_BUCKET} --key scripts/
aws s3api put-object --bucket ${MODEL_BUCKET} --key model/latest/

aws s3 cp starter/data/creditcard.csv s3://${DATA_BUCKET}/raw/creditcard.csv
aws s3 cp starter/fraud_detection_pipeline/fraud_detector_model_trainer.py s3://${DATA_BUCKET}/scripts/fraud_detector_model_trainer.py
log "STEP 1/7 completed"

### 2) Create IAM roles (Glue, Lambda, ECS)
log "STEP 2/7: Creating/reusing IAM roles and applying policies"

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

if ! aws iam get-role --role-name ${PROJECT}-glue-role >/dev/null 2>&1; then
  aws iam create-role --role-name ${PROJECT}-glue-role --assume-role-policy-document file://trust-glue.json
fi

if ! aws iam get-role --role-name ${PROJECT}-lambda-role >/dev/null 2>&1; then
  aws iam create-role --role-name ${PROJECT}-lambda-role --assume-role-policy-document file://trust-lambda.json
fi

if ! aws iam get-role --role-name ${PROJECT}-ecs-exec-role >/dev/null 2>&1; then
  aws iam create-role --role-name ${PROJECT}-ecs-exec-role --assume-role-policy-document file://trust-ecs-task.json
fi

if ! aws iam get-role --role-name ${PROJECT}-ecs-task-role >/dev/null 2>&1; then
  aws iam create-role --role-name ${PROJECT}-ecs-task-role --assume-role-policy-document file://trust-ecs-task.json
fi

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
log "STEP 2/7 completed"

### 3) Build/push API image to ECR
log "STEP 3/7: Building API image and pushing to ECR"
aws ecr create-repository --repository-name ${ECR_REPO} --region ${AWS_REGION} || true

aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

docker buildx create --name fraud-builder --use >/dev/null 2>&1 || docker buildx use fraud-builder
docker buildx build \
  --platform linux/amd64 \
  -t ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG} \
  --push \
  starter/fraud_detection_api
log "STEP 3/7 completed"

### 4) Deploy ECS Fargate + ALB
log "STEP 4/7: Deploying/reconciling ECS service and ALB"
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SUBNET_1=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=${VPC_ID} --query 'Subnets[0].SubnetId' --output text)
SUBNET_2=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=${VPC_ID} --query 'Subnets[1].SubnetId' --output text)

ALB_SG_ID=$(aws ec2 describe-security-groups \
  --filters Name=vpc-id,Values=${VPC_ID} Name=group-name,Values=${PROJECT}-alb-sg \
  --query 'SecurityGroups[0].GroupId' --output text)

if [ "${ALB_SG_ID}" = "None" ] || [ -z "${ALB_SG_ID}" ]; then
  ALB_SG_ID=$(aws ec2 create-security-group \
    --group-name ${PROJECT}-alb-sg \
    --description "ALB ingress on 80" \
    --vpc-id ${VPC_ID} \
    --query 'GroupId' --output text)
fi

aws ec2 authorize-security-group-ingress \
  --group-id ${ALB_SG_ID} \
  --ip-permissions IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges='[{CidrIp=0.0.0.0/0,Description="Allow HTTP"}]' || true

ECS_SG_ID=$(aws ec2 describe-security-groups \
  --filters Name=vpc-id,Values=${VPC_ID} Name=group-name,Values=${PROJECT}-ecs-sg \
  --query 'SecurityGroups[0].GroupId' --output text)

if [ "${ECS_SG_ID}" = "None" ] || [ -z "${ECS_SG_ID}" ]; then
  ECS_SG_ID=$(aws ec2 create-security-group \
    --group-name ${PROJECT}-ecs-sg \
    --description "ECS ingress from ALB on 8000" \
    --vpc-id ${VPC_ID} \
    --query 'GroupId' --output text)
fi

aws ec2 authorize-security-group-ingress \
  --group-id ${ECS_SG_ID} \
  --ip-permissions IpProtocol=tcp,FromPort=8000,ToPort=8000,UserIdGroupPairs='[{GroupId='${ALB_SG_ID}',Description="Allow ALB to ECS"}]' || true

if ! aws ecs describe-clusters --clusters ${ECS_CLUSTER} --query 'clusters[0].clusterArn' --output text | grep -q '^arn:'; then
  aws ecs create-cluster --cluster-name ${ECS_CLUSTER}
fi

cat > taskdef.json << EOF
{
  "family": "${TASK_FAMILY}",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "runtimePlatform": {
    "operatingSystemFamily": "LINUX",
    "cpuArchitecture": "X86_64"
  },
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
        {"name":"MODEL_S3_URI","value":"s3://${MODEL_BUCKET}/model/latest/"}
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

ALB_ARN=$(aws elbv2 describe-load-balancers --names ${ALB_NAME} --query 'LoadBalancers[0].LoadBalancerArn' --output text 2>/dev/null)

if [ "${ALB_ARN}" = "None" ] || [ -z "${ALB_ARN}" ]; then
  aws elbv2 create-load-balancer \
    --name ${ALB_NAME} \
    --subnets ${SUBNET_1} ${SUBNET_2} \
    --security-groups ${ALB_SG_ID} \
    --scheme internet-facing \
    --type application
fi

ALB_ARN=$(aws elbv2 describe-load-balancers --names ${ALB_NAME} --query 'LoadBalancers[0].LoadBalancerArn' --output text)
aws elbv2 set-security-groups --load-balancer-arn ${ALB_ARN} --security-groups ${ALB_SG_ID}
aws elbv2 set-subnets --load-balancer-arn ${ALB_ARN} --subnets ${SUBNET_1} ${SUBNET_2}

TG_ARN=$(aws elbv2 describe-target-groups --names ${TG_NAME} --query 'TargetGroups[0].TargetGroupArn' --output text 2>/dev/null || true)

if [ "${TG_ARN}" = "None" ] || [ -z "${TG_ARN}" ]; then
  aws elbv2 create-target-group \
    --name ${TG_NAME} \
    --protocol HTTP \
    --port 8000 \
    --target-type ip \
    --vpc-id ${VPC_ID}
fi

TG_ARN=$(aws elbv2 describe-target-groups --names ${TG_NAME} --query 'TargetGroups[0].TargetGroupArn' --output text)

LISTENER_ARN=$(aws elbv2 describe-listeners --load-balancer-arn ${ALB_ARN} --query "Listeners[?Port==\`80\`].ListenerArn | [0]" --output text)

if [ "${LISTENER_ARN}" = "None" ] || [ -z "${LISTENER_ARN}" ]; then
  aws elbv2 create-listener \
    --load-balancer-arn ${ALB_ARN} \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=${TG_ARN}
else
  aws elbv2 modify-listener \
    --listener-arn ${LISTENER_ARN} \
    --default-actions Type=forward,TargetGroupArn=${TG_ARN}
fi

if ! aws ecs describe-services --cluster ${ECS_CLUSTER} --services ${ECS_SERVICE} --query 'services[0].serviceArn' --output text | grep -q '^arn:'; then
  aws ecs create-service \
    --cluster ${ECS_CLUSTER} \
    --service-name ${ECS_SERVICE} \
    --task-definition ${TASK_FAMILY} \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[${SUBNET_1},${SUBNET_2}],securityGroups=[${ECS_SG_ID}],assignPublicIp=ENABLED}" \
    --load-balancers "targetGroupArn=${TG_ARN},containerName=fraud-api,containerPort=8000"
else
  aws ecs update-service \
    --cluster ${ECS_CLUSTER} \
    --service ${ECS_SERVICE} \
    --task-definition ${TASK_FAMILY} \
    --desired-count 1 \
    --force-new-deployment
fi
  log "STEP 4/7 completed"

### 5) Create Glue training job
  log "STEP 5/7: Creating/updating Glue job and starting a run"

if ! aws glue get-job --job-name ${GLUE_JOB_NAME} >/dev/null 2>&1; then
  aws glue create-job \
    --name ${GLUE_JOB_NAME} \
    --role arn:aws:iam::${ACCOUNT_ID}:role/${PROJECT}-glue-role \
    --command Name=glueetl,ScriptLocation=s3://${DATA_BUCKET}/scripts/fraud_detector_model_trainer.py,PythonVersion=3 \
    --glue-version 4.0 \
    --worker-type G.1X \
    --number-of-workers 2 \
    --default-arguments "{\"--DATA_URL\":\"s3://${DATA_BUCKET}/raw/creditcard.csv\",\"--MODEL_SAVE_PATH\":\"s3://${MODEL_BUCKET}/model/latest/\",\"--JOB_NAME\":\"${GLUE_JOB_NAME}\"}"
else
  aws glue update-job \
    --job-name ${GLUE_JOB_NAME} \
    --job-update "{\"Role\":\"arn:aws:iam::${ACCOUNT_ID}:role/${PROJECT}-glue-role\",\"Command\":{\"Name\":\"glueetl\",\"ScriptLocation\":\"s3://${DATA_BUCKET}/scripts/fraud_detector_model_trainer.py\",\"PythonVersion\":\"3\"},\"GlueVersion\":\"4.0\",\"WorkerType\":\"G.1X\",\"NumberOfWorkers\":2,\"DefaultArguments\":{\"--DATA_URL\":\"s3://${DATA_BUCKET}/raw/creditcard.csv\",\"--MODEL_SAVE_PATH\":\"s3://${MODEL_BUCKET}/model/latest/\",\"--JOB_NAME\":\"${GLUE_JOB_NAME}\"}}"
fi

GLUE_LAST_STATE=$(aws glue get-job-runs --job-name ${GLUE_JOB_NAME} --max-results 1 --query 'JobRuns[0].JobRunState' --output text 2>/dev/null || true)
if [ "${GLUE_LAST_STATE}" = "STARTING" ] || [ "${GLUE_LAST_STATE}" = "RUNNING" ] || [ "${GLUE_LAST_STATE}" = "STOPPING" ]; then
  log "Glue job ${GLUE_JOB_NAME} already active (state=${GLUE_LAST_STATE}); skipping start-job-run"
else
  GLUE_JOB_RUN_ID=$(aws glue start-job-run --job-name ${GLUE_JOB_NAME} --query 'JobRunId' --output text)
  log "Started Glue job run: ${GLUE_JOB_RUN_ID}"
fi
log "STEP 5/7 completed"

  ### 6) Create Lambda trigger for retrain files
log "STEP 6/7: Creating/updating Lambda and wiring S3 trigger"

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

if ! aws lambda get-function --function-name ${LAMBDA_NAME} >/dev/null 2>&1; then
  aws lambda create-function \
    --function-name ${LAMBDA_NAME} \
    --runtime python3.12 \
    --handler lambda_function.lambda_handler \
    --zip-file fileb://lambda.zip \
    --role arn:aws:iam::${ACCOUNT_ID}:role/${PROJECT}-lambda-role \
    --environment "Variables={GLUE_JOB_NAME=${GLUE_JOB_NAME}}"
else
  wait_for_lambda_update_ready ${LAMBDA_NAME} 30 5
  aws lambda update-function-code \
    --function-name ${LAMBDA_NAME} \
    --zip-file fileb://lambda.zip
  safe_lambda_update_configuration ${LAMBDA_NAME} \
    --runtime python3.12 \
    --handler lambda_function.lambda_handler \
    --role arn:aws:iam::${ACCOUNT_ID}:role/${PROJECT}-lambda-role \
    --environment "Variables={GLUE_JOB_NAME=${GLUE_JOB_NAME}}"
fi

LAMBDA_ARN=$(aws lambda get-function --function-name ${LAMBDA_NAME} --query 'Configuration.FunctionArn' --output text)

aws lambda remove-permission --function-name ${LAMBDA_NAME} --statement-id s3invoke >/dev/null 2>&1 || true
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
log "STEP 6/7 completed"

  ### 7) Validate end-to-end
  log "STEP 7/7: Running end-to-end validation"
  aws s3 cp starter/data/creditcard.csv s3://${DATA_BUCKET}/retrain/retrain_$(date +%s).csv

  wait_for_ecs_service_stable ${ECS_CLUSTER} ${ECS_SERVICE} 30 20

  ALB_DNS=$(aws elbv2 describe-load-balancers --names ${ALB_NAME} --query 'LoadBalancers[0].DNSName' --output text)
curl -X POST "http://${ALB_DNS}/predict/" -H "Content-Type: application/json" -d '{"features":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}'
  log "STEP 7/7 completed"


# 1) Edit placeholders in the policy file first:
# <AWS_REGION> <ACCOUNT_ID> <GLUE_JOB_NAME> <DATA_BUCKET> <RETRAIN_PREFIX> <ARCHIVE_PREFIX>

# 2) Attach as inline policy to your Lambda execution role
aws iam put-role-policy \
  --role-name  ${PROJECT}-lambda-role \
  --policy-name refresh-function-inline-policy \
  --policy-document file://starter/refresh_function/lambda_iam_policy.json

# 3) Set Lambda env vars used by refresh_function.py
safe_lambda_update_configuration ${LAMBDA_NAME} \
  --environment "Variables={GLUE_JOB_NAME=${GLUE_JOB_NAME},MODEL_SAVE_PATH=s3://${MODEL_BUCKET}/model/latest/,ARCHIVE_PREFIX=archive}"

log "Runbook completed successfully."