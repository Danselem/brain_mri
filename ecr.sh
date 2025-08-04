#!/bin/bash

set -euo pipefail

# Load environment variables
if [ ! -f .env ]; then
  echo ".env file not found!"
  exit 1
fi
set -a
source .env
set +a

# Get AWS Account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

echo "Using ECR URI: $ECR_URI"

# Authenticate Docker to ECR
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_URI"

# Create ECR repository if it doesn't exist
if ! aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$AWS_REGION" > /dev/null 2>&1; then
  echo "Creating ECR repository: $ECR_REPO_NAME"
  aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$AWS_REGION"
fi

# Ensure buildx builder exists and is in use
if ! docker buildx inspect multi-builder > /dev/null 2>&1; then
  docker buildx create --name multi-builder --use
fi
docker buildx use multi-builder

# Build and push multi-platform image (amd64 + arm64)
docker buildx build --platform linux/amd64 \
  -t "${ECR_URI}:${IMAGE_TAG}" \
  --push .

echo "Multi-arch image pushed to: ${ECR_URI}:${IMAGE_TAG}"
