# Deployment Guide

Complete guide for deploying the Voice Authenticity Detection API to production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [AWS Deployment](#aws-deployment)
5. [Google Cloud Platform](#google-cloud-platform)
6. [Azure Deployment](#azure-deployment)
7. [Monitoring & Logging](#monitoring--logging)
8. [Security Best Practices](#security-best-practices)
9. [Scaling](#scaling)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- Python 3.10 or higher
- Docker 20.10+ (for containerized deployment)
- 2GB RAM minimum (4GB recommended)
- 10GB disk space

### Required Tools
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3-pip python3-venv ffmpeg git

# macOS
brew install python@3.10 ffmpeg git

# Verify installations
python3 --version
docker --version
ffmpeg -version
```

---

## Local Development

### 1. Clone Repository

```bash
git clone https://github.com/KSingh0405/Voice_detection_api.git
cd voice-auth-api
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment

Create `.env` file:
```bash
cat > .env << EOF
API_KEYS=dev_key_123,test_key_456
LOG_LEVEL=DEBUG
MODEL_PATH=models/weights
EOF
```

### 5. Run Development Server

```bash
# With auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Access API
curl http://localhost:8000/health
```

### 6. Run Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

---

## Docker Deployment

### Single Container

```bash
# Build image
docker build -t voice-auth-api:latest .

# Run container
docker run -d \
  --name voice-auth-api \
  -p 8000:8000 \
  -e API_KEYS="prod_key_123,prod_key_456" \
  -v $(pwd)/models/weights:/app/models/weights \
  -v $(pwd)/logs:/app/logs \
  --restart unless-stopped \
  voice-auth-api:latest

# View logs
docker logs -f voice-auth-api

# Stop container
docker stop voice-auth-api
docker rm voice-auth-api
```

### Docker Compose (Recommended)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

### Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    container_name: voice-auth-api-prod
    ports:
      - "8000:8000"
    environment:
      - API_KEYS=${API_KEYS}
      - LOG_LEVEL=INFO
    volumes:
      - ./models/weights:/app/models/weights:ro
      - ./logs:/app/logs
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  nginx:
    image: nginx:alpine
    container_name: voice-auth-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
    restart: always
```

### NGINX Configuration

Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }

    server {
        listen 80;
        server_name api.yourdomain.com;

        location / {
            return 301 https://$server_name$request_uri;
        }
    }

    server {
        listen 443 ssl http2;
        server_name api.yourdomain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        client_max_body_size 10M;

        location / {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
    }
}
```

---

## AWS Deployment

### Option 1: EC2 with Docker

#### Step 1: Launch EC2 Instance

```bash
# Launch Ubuntu 22.04 instance (t3.medium or larger)
# Configure security group:
# - Allow SSH (22) from your IP
# - Allow HTTP (80) from 0.0.0.0/0
# - Allow HTTPS (443) from 0.0.0.0/0
# - Allow Custom TCP (8000) from 0.0.0.0/0 (optional, for direct API access)
```

#### Step 2: Connect and Setup

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-public-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Logout and login again for group changes
exit
```

#### Step 3: Deploy Application

```bash
# Clone repository
git clone https://github.com/yourusername/voice-auth-api.git
cd voice-auth-api

# Set environment variables
export API_KEYS="your_production_key_here"
echo "API_KEYS=$API_KEYS" > .env

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Verify
curl http://localhost:8000/health
```

#### Step 4: Setup SSL with Let's Encrypt

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d api.yourdomain.com

# Auto-renewal is configured automatically
sudo certbot renew --dry-run
```

### Option 2: ECS Fargate

#### Step 1: Create ECR Repository

```bash
# Create repository
aws ecr create-repository --repository-name voice-auth-api

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t voice-auth-api .
docker tag voice-auth-api:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/voice-auth-api:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/voice-auth-api:latest
```

#### Step 2: Create Task Definition

Create `task-definition.json`:

```json
{
  "family": "voice-auth-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "voice-auth-api",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/voice-auth-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "API_KEYS",
          "value": "your_production_key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/voice-auth-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create ECS cluster
aws ecs create-cluster --cluster-name voice-auth-cluster

# Create service with ALB
aws ecs create-service \
  --cluster voice-auth-cluster \
  --service-name voice-auth-service \
  --task-definition voice-auth-api \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=voice-auth-api,containerPort=8000"
```

### Option 3: Lambda with API Gateway

For serverless deployment (lower traffic):

```bash
# Install serverless framework
npm install -g serverless

# Create serverless.yml
# Deploy
serverless deploy
```

---

## Google Cloud Platform

### Cloud Run Deployment

#### Step 1: Build and Push to GCR

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Build image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/voice-auth-api

# Or build locally and push
docker build -t gcr.io/YOUR_PROJECT_ID/voice-auth-api .
docker push gcr.io/YOUR_PROJECT_ID/voice-auth-api
```

#### Step 2: Deploy to Cloud Run

```bash
gcloud run deploy voice-auth-api \
  --image gcr.io/YOUR_PROJECT_ID/voice-auth-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars API_KEYS="your_production_key" \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10 \
  --min-instances 1

# Get service URL
gcloud run services describe voice-auth-api --region us-central1 --format 'value(status.url)'
```

### GKE Deployment (Kubernetes)

```bash
# Create cluster
gcloud container clusters create voice-auth-cluster \
  --num-nodes 3 \
  --machine-type n1-standard-2 \
  --region us-central1

# Deploy application
kubectl apply -f k8s/deployment.yml
kubectl apply -f k8s/service.yml
kubectl apply -f k8s/ingress.yml
```

---

## Azure Deployment

### Azure Container Instances

```bash
# Create resource group
az group create --name voice-auth-rg --location eastus

# Create container registry
az acr create --resource-group voice-auth-rg --name voiceauthreg --sku Basic

# Build and push
az acr build --registry voiceauthreg --image voice-auth-api .

# Deploy container
az container create \
  --resource-group voice-auth-rg \
  --name voice-auth-api \
  --image voiceauthreg.azurecr.io/voice-auth-api \
  --dns-name-label voice-auth \
  --ports 8000 \
  --environment-variables API_KEYS="your_production_key" \
  --cpu 2 \
  --memory 4
```

### Azure App Service

```bash
# Create App Service plan
az appservice plan create \
  --name voice-auth-plan \
  --resource-group voice-auth-rg \
  --is-linux \
  --sku B2

# Create web app
az webapp create \
  --resource-group voice-auth-rg \
  --plan voice-auth-plan \
  --name voice-auth-api \
  --deployment-container-image-name voiceauthreg.azurecr.io/voice-auth-api

# Configure environment
az webapp config appsettings set \
  --resource-group voice-auth-rg \
  --name voice-auth-api \
  --settings API_KEYS="your_production_key"
```

---

## Monitoring & Logging

### Application Monitoring

Install monitoring tools:

```bash
pip install prometheus-client python-json-logger
```

Add metrics endpoint in `main.py`:

```python
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

# Metrics
request_count = Counter('api_requests_total', 'Total API requests')
request_duration = Histogram('api_request_duration_seconds', 'Request duration')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Logging Configuration

Configure structured logging:

```python
import logging
import json_log_formatter

formatter = json_log_formatter.JSONFormatter()
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

### CloudWatch (AWS)

```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i amazon-cloudwatch-agent.deb

# Configure and start
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -s \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/config.json
```

---

## Security Best Practices

### 1. API Key Management

```bash
# Use AWS Secrets Manager
aws secretsmanager create-secret \
  --name voice-auth-api-keys \
  --secret-string '{"keys":["key1","key2"]}'

# Retrieve in application
import boto3
client = boto3.client('secretsmanager')
response = client.get_secret_value(SecretId='voice-auth-api-keys')
```

### 2. Rate Limiting

Install and configure rate limiting:

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/detect")
@limiter.limit("100/hour")
async def detect(...):
    ...
```

### 3. HTTPS Only

```python
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

app.add_middleware(HTTPSRedirectMiddleware)
```

### 4. CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

---

## Scaling

### Horizontal Scaling

#### Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml voice-auth

# Scale service
docker service scale voice-auth_api=5
```

#### Kubernetes

```yaml
# deployment.yml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: voice-auth-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: voice-auth-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Load Balancing

Configure NGINX load balancer:

```nginx
upstream api_backend {
    least_conn;
    server api1:8000 weight=1;
    server api2:8000 weight=1;
    server api3:8000 weight=1;
}
```

---

## Troubleshooting

### Common Issues

**Issue**: Container won't start
```bash
# Check logs
docker logs voice-auth-api

# Check if port is in use
sudo netstat -tlnp | grep 8000

# Verify dependencies
docker run -it voice-auth-api:latest /bin/bash
pip list
```

**Issue**: Out of memory
```bash
# Increase Docker memory
# Docker Desktop: Settings > Resources > Memory

# Check memory usage
docker stats voice-auth-api

# Optimize model loading
# Use model quantization or smaller model
```

**Issue**: Slow response times
```bash
# Profile the application
pip install py-spy
py-spy record -o profile.svg -- python main.py

# Add caching for feature extraction
# Use Redis for result caching
```

---

## Performance Optimization

### 1. Enable Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def extract_features_cached(audio_hash):
    return extract_features(audio)
```

### 2. Use Gunicorn with Workers

```bash
pip install gunicorn

gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

### 3. Database for API Keys

```python
# Use Redis for fast lookups
import redis
r = redis.Redis(host='localhost', port=6379)

def verify_api_key(key):
    return r.exists(f"api_key:{key}")
```

---

## Backup & Recovery

### Backup Model Weights

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR
cp -r /app/models/weights $BACKUP_DIR/
aws s3 sync $BACKUP_DIR s3://your-backup-bucket/
```

### Database Backup (if using)

```bash
# Backup PostgreSQL
pg_dump -U user database > backup.sql

# Restore
psql -U user database < backup.sql
```

---

## Support & Maintenance

For production support:
- Monitor error rates and latency
- Set up alerts for failures
- Regular security updates
- Model retraining schedule
- Capacity planning

---

**Next Steps**: After deployment, configure monitoring, set up automated backups, and establish an incident response plan.
