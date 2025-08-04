#!/bin/bash

set -euo pipefail

# Load environment variables
if [ ! -f .env ]; then
  echo ".env file not found! Please include AWS_REGION, ECR_REPO_NAME, IMAGE_TAG."
  exit 1
fi

set -a
source .env
set +a

echo "AWS_REGION=$AWS_REGION"
echo "ECR_REPO_NAME=$ECR_REPO_NAME"
echo "IMAGE_TAG=$IMAGE_TAG"
echo

CLUSTER_NAME="tumor-classifier-cluster"
SERVICE_NAME="tumor-classifier-service"

echo "CLUSTER_NAME=$CLUSTER_NAME"
echo "SERVICE_NAME=$SERVICE_NAME"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --no-cli-pager)

VPC_ID=$(aws ec2 describe-vpcs \
  --region "$AWS_REGION" \
  --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' --output text --no-cli-pager)

if [ "$VPC_ID" == "None" ] || [ -z "$VPC_ID" ]; then
  echo "No default VPC found in region $AWS_REGION."
  exit 1
fi

echo "VPC_ID=$VPC_ID"

SUBNET_IDS=$(aws ec2 describe-subnets \
  --region "$AWS_REGION" \
  --filters Name=vpc-id,Values="$VPC_ID" Name=map-public-ip-on-launch,Values=true \
  --query 'Subnets[].SubnetId' --output text --no-cli-pager)

if [ -z "$SUBNET_IDS" ]; then
  echo "No public subnets found in VPC $VPC_ID."
  exit 1
fi

echo "SUBNET_IDS=$SUBNET_IDS"
echo

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"
echo "ECR_URI: $ECR_URI"
echo

# 1. Create ECS Cluster if needed
if aws ecs describe-clusters --clusters "$CLUSTER_NAME" --query 'clusters[0].status' --output text --no-cli-pager 2>/dev/null | grep -q ACTIVE; then
  echo "ECS Cluster $CLUSTER_NAME exists."
else
  echo "Creating ECS Cluster: $CLUSTER_NAME"
  aws ecs create-cluster --cluster-name "$CLUSTER_NAME" --no-cli-pager >/dev/null
fi

# 2. Create IAM Role if needed
ROLE_NAME="ecsTaskExecutionRole"
if aws iam get-role --role-name "$ROLE_NAME" --no-cli-pager >/dev/null 2>&1; then
  echo "IAM Role $ROLE_NAME exists."
else
  echo "Creating IAM Role $ROLE_NAME"
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document file://<(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "ecs-tasks.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
EOF
) --no-cli-pager >/dev/null
  aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
    --no-cli-pager
  echo "Waiting for IAM role propagation..."
  sleep 10
fi

# 3. Create Security Group allowing inbound TCP 9696
SG_NAME="${CLUSTER_NAME}-http-sg"
SG_ID=$(aws ec2 describe-security-groups \
  --filters Name=group-name,Values="$SG_NAME" \
  --query 'SecurityGroups[0].GroupId' --output text --no-cli-pager 2>/dev/null || echo "None")

if [ "$SG_ID" == "None" ]; then
  echo "Creating Security Group: $SG_NAME"
  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "Allow TCP port 9696 traffic" \
    --vpc-id "$VPC_ID" --query GroupId --output text --no-cli-pager)
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp --port 9696 --cidr 0.0.0.0/0 \
    --no-cli-pager >/dev/null
  echo "Security Group created: $SG_ID"
else
  echo "Security Group $SG_NAME exists: $SG_ID"
fi

# 4. Register Task Definition with port 9696 & AWS Logs config
TASK_DEF_NAME="${SERVICE_NAME}-task"
echo "Registering ECS Task Definition $TASK_DEF_NAME..."

TASK_DEF_ARN=$(aws ecs register-task-definition \
  --family "$TASK_DEF_NAME" \
  --network-mode "awsvpc" \
  --requires-compatibilities "FARGATE" \
  --cpu "512" \
  --memory "1024" \
  --execution-role-arn "arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}" \
  --container-definitions "$(cat <<EOF
[
  {
    "name": "${SERVICE_NAME}-container",
    "image": "${ECR_URI}",
    "portMappings": [
      {
        "containerPort": 9696,
        "hostPort": 9696,
        "protocol": "tcp"
      }
    ],
    "essential": true,
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/${SERVICE_NAME}",
        "awslogs-region": "${AWS_REGION}",
        "awslogs-stream-prefix": "ecs"
      }
    }
  }
]
EOF
)" --query 'taskDefinition.taskDefinitionArn' --output text --no-cli-pager)

echo "Task Definition ARN: $TASK_DEF_ARN"
echo

# 5. Create or Update ECS Service
SERVICE_EXISTS=$(aws ecs describe-services \
  --cluster "$CLUSTER_NAME" \
  --services "$SERVICE_NAME" \
  --query 'services[0].status' --output text --no-cli-pager 2>/dev/null || echo "NONE")

# Convert space-separated SUBNET_IDS to JSON array of quoted strings
read -ra SUBNET_ARRAY <<< "$SUBNET_IDS"

# Build JSON array string correctly, with commas between quoted subnet IDs
SUBNETS_JSON="["
for subnet in "${SUBNET_ARRAY[@]}"; do
  # Trim any whitespace (optional)
  subnet=$(echo "$subnet" | xargs)
  SUBNETS_JSON+="\"${subnet}\","
done
SUBNETS_JSON="${SUBNETS_JSON%,}]"  # Remove trailing comma and close array

echo "Using subnets JSON: $SUBNETS_JSON"


echo "Using subnets JSON: $SUBNETS_JSON"

if [ "$SERVICE_EXISTS" == "ACTIVE" ]; then
  echo "Updating ECS Service..."
  aws ecs update-service \
    --cluster "$CLUSTER_NAME" \
    --service "$SERVICE_NAME" \
    --task-definition "$TASK_DEF_ARN" \
    --no-cli-pager >/dev/null
else
  echo "Creating ECS Service..."
  aws ecs create-service \
    --cluster "$CLUSTER_NAME" \
    --service-name "$SERVICE_NAME" \
    --task-definition "$TASK_DEF_ARN" \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=$SUBNETS_JSON,securityGroups=[\"$SG_ID\"],assignPublicIp=ENABLED}" \
    --region "$AWS_REGION" \
    --output text --no-cli-pager >/dev/null
fi

echo
echo "Waiting for ECS service to stabilize (30s)..."
sleep 30

# 6. Show Service Details
echo
echo "ECS Service Status:"
aws ecs describe-services \
  --cluster "$CLUSTER_NAME" \
  --services "$SERVICE_NAME" \
  --query 'services[0].[serviceName,status,desiredCount,runningCount]' \
  --output table --no-cli-pager

echo
echo "Running Task(s):"
TASKS=$(aws ecs list-tasks \
  --cluster "$CLUSTER_NAME" \
  --service-name "$SERVICE_NAME" \
  --query 'taskArns' --output text --no-cli-pager)
echo "$TASKS"

echo
echo "Public IP(s) of running task(s):"
for TASK_ARN in $TASKS; do
  ENI_ID=$(aws ecs describe-tasks \
    --cluster "$CLUSTER_NAME" \
    --tasks "$TASK_ARN" \
    --query 'tasks[0].attachments[].details[?name==`networkInterfaceId`].value | [0]' \
    --output text --no-cli-pager)

  if [ -z "$ENI_ID" ] || [ "$ENI_ID" == "None" ]; then
    echo "Warning: No network interface ID found for task $TASK_ARN"
    continue
  fi

  PUBLIC_IP=$(aws ec2 describe-network-interfaces \
    --network-interface-ids "$ENI_ID" \
    --query 'NetworkInterfaces[0].Association.PublicIp' \
    --output text --no-cli-pager)

  echo "Task: $TASK_ARN | Public IP: $PUBLIC_IP"
done


echo
echo "✅ Deployment complete. Visit your app using the public IP(s) above on port 9696."
