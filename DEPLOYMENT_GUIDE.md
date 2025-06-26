# OpenInferencev2 Deployment Guide

## Production Deployment Guide

This guide covers enterprise-grade deployment of OpenInferencev2 for production environments.

## Prerequisites

### Hardware Requirements

#### Minimum Requirements
- **CPU**: 8 cores, 2.5GHz+
- **RAM**: 32GB system memory
- **GPU**: RTX 3080 or better (8GB+ VRAM)
- **Storage**: 100GB SSD storage
- **Network**: 1Gbps+ network connection

#### Recommended Requirements
- **CPU**: 16+ cores, 3.0GHz+ (Intel Xeon or AMD EPYC)
- **RAM**: 128GB+ system memory
- **GPU**: RTX 4090 or A100 (24GB+ VRAM)
- **Storage**: 500GB+ NVMe SSD
- **Network**: 10Gbps+ network connection

### Software Requirements

#### Operating System
- **Primary**: Ubuntu 20.04/22.04 LTS
- **Alternative**: RHEL 8/9, CentOS Stream 8/9
- **Container**: Docker 20.10+ or Podman 3.0+

#### Dependencies
- **CUDA**: 11.8+ or 12.0+ (matching PyTorch version)
- **Python**: 3.8+ (recommended: 3.11+)
- **NVIDIA Driver**: 520+ (recommended: latest)
- **Docker**: 20.10+ (for containerized deployment)
- **Kubernetes**: 1.25+ (for orchestrated deployment)

## Installation Methods

### Method 1: Direct Installation

#### Step 1: System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers
sudo apt install nvidia-driver-535 nvidia-utils-535

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Verify installation
nvidia-smi
nvcc --version
```

#### Step 2: Python Environment

```bash
# Install Python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11 python3.11-venv python3.11-dev

# Create virtual environment
python3.11 -m venv openinferencev2-env
source openinferencev2-env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Step 3: Install OpenInferencev2

```bash
# Clone repository
git clone https://github.com/llamasearchai/OpenInferencev2.git
cd OpenInferencev2

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

# Verify installation
python -c "import openinferencev2; print(openinferencev2.__version__)"
```

#### Step 4: Configuration

```bash
# Create configuration directory
sudo mkdir -p /etc/openinferencev2
sudo chown $USER:$USER /etc/openinferencev2

# Copy configuration template
cp config/production.json /etc/openinferencev2/config.json

# Edit configuration
nano /etc/openinferencev2/config.json
```

#### Step 5: System Service

```bash
# Create systemd service
sudo nano /etc/systemd/system/openinferencev2.service
```

```ini
[Unit]
Description=OpenInferencev2 Inference Engine
After=network.target nvidia-persistenced.service

[Service]
Type=simple
User=openinferencev2
Group=openinferencev2
WorkingDirectory=/opt/openinferencev2
ExecStart=/opt/openinferencev2/venv/bin/python -m src.cli.main --model /opt/models/llama2-7b --config /etc/openinferencev2/config.json
Restart=on-failure
RestartSec=5
Environment=CUDA_VISIBLE_DEVICES=0,1
Environment=PYTHONPATH=/opt/openinferencev2

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable openinferencev2
sudo systemctl start openinferencev2
```

### Method 2: Docker Deployment

#### Step 1: Build Docker Image

```bash
# Build production image
docker build -f Dockerfile -t openinferencev2:latest .

# Or use multi-stage build for smaller image
docker build -f Dockerfile.prod -t openinferencev2:prod .
```

#### Step 2: Run Container

```bash
# Run with GPU support
docker run -d \
  --name openinferencev2 \
  --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  -v /opt/models:/models:ro \
  -v /opt/config:/config:ro \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  openinferencev2:latest \
  --model /models/llama2-7b \
  --config /config/production.json
```

#### Step 3: Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  openinferencev2:
    image: openinferencev2:latest
    build: .
    ports:
      - "8000:8000"
    volumes:
      - /opt/models:/models:ro
      - /opt/config:/config:ro
      - /var/log/openinferencev2:/logs
    environment:
      - CUDA_VISIBLE_DEVICES=0,1
      - PYTHONPATH=/app
    command: >
      --model /models/llama2-7b
      --config /config/production.json
      --log-level INFO
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Method 3: Kubernetes Deployment

#### Step 1: Create Namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: openinferencev2
```

#### Step 2: ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: openinferencev2-config
  namespace: openinferencev2
data:
  config.json: |
    {
      "num_gpus": 2,
      "max_batch_size": 32,
      "max_sequence_length": 4096,
      "use_fp16": true,
      "use_flash_attention": true,
      "kv_cache_size_gb": 16,
      "tensor_parallel_size": 2,
      "pipeline_parallel_size": 1,
      "monitoring": {
        "enable_prometheus": true,
        "metrics_port": 9090
      }
    }
```

#### Step 3: Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openinferencev2
  namespace: openinferencev2
spec:
  replicas: 2
  selector:
    matchLabels:
      app: openinferencev2
  template:
    metadata:
      labels:
        app: openinferencev2
    spec:
      containers:
      - name: openinferencev2
        image: openinferencev2:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1"
        - name: PYTHONPATH
          value: "/app"
        volumeMounts:
        - name: config
          mountPath: /config
        - name: models
          mountPath: /models
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: "2"
          limits:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: openinferencev2-config
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      nodeSelector:
        accelerator: nvidia-tesla-v100
```

#### Step 4: Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: openinferencev2-service
  namespace: openinferencev2
spec:
  selector:
    app: openinferencev2
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
```

## Configuration

### Production Configuration Template

```json
{
  "model": {
    "path": "/models/llama2-7b",
    "type": "llama",
    "revision": "main"
  },
  "engine": {
    "num_gpus": 2,
    "max_batch_size": 32,
    "max_sequence_length": 4096,
    "tensor_parallel_size": 2,
    "pipeline_parallel_size": 1,
    "use_fp16": true,
    "use_flash_attention": true,
    "use_cuda_graphs": true,
    "quantization": null,
    "kv_cache_size_gb": 16.0,
    "max_memory_per_gpu_gb": 22.0
  },
  "serving": {
    "host": "0.0.0.0",
    "port": 8000,
    "max_concurrent_requests": 1000,
    "request_timeout": 60,
    "enable_cors": true,
    "api_keys": []
  },
  "monitoring": {
    "enable_prometheus": true,
    "metrics_port": 9090,
    "log_level": "INFO",
    "enable_health_checks": true,
    "health_check_interval": 30
  },
  "optimization": {
    "enable_dynamic_batching": true,
    "max_wait_time": 0.1,
    "preferred_batch_size": 16,
    "enable_request_preprocessing": true,
    "enable_response_caching": false
  },
  "security": {
    "enable_rate_limiting": true,
    "rate_limit_per_minute": 100,
    "enable_request_validation": true,
    "max_prompt_length": 8192,
    "max_response_length": 4096
  }
}
```

### Environment Variables

```bash
# Core settings
export OPENINFERENCEV2_MODEL_PATH="/models/llama2-7b"
export OPENINFERENCEV2_CONFIG_PATH="/etc/openinferencev2/config.json"
export OPENINFERENCEV2_LOG_LEVEL="INFO"

# GPU settings
export CUDA_VISIBLE_DEVICES="0,1"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# Memory settings
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING="0"

# Performance settings
export OMP_NUM_THREADS="8"
export MKL_NUM_THREADS="8"
```

## Performance Tuning

### GPU Optimization

#### Memory Management
```json
{
  "memory_optimization": {
    "kv_cache_size_gb": 16.0,
    "max_memory_per_gpu_gb": 22.0,
    "memory_pool_size": "80%",
    "enable_memory_defragmentation": true,
    "garbage_collection_threshold": 0.8
  }
}
```

#### Compute Optimization
```json
{
  "compute_optimization": {
    "use_fp16": true,
    "use_flash_attention": true,
    "use_cuda_graphs": true,
    "enable_kernel_fusion": true,
    "tensor_parallel_size": 2,
    "pipeline_parallel_size": 1
  }
}
```

### Batch Processing

#### Dynamic Batching
```json
{
  "batching": {
    "enable_dynamic_batching": true,
    "max_batch_size": 32,
    "preferred_batch_size": 16,
    "max_wait_time": 0.1,
    "padding_strategy": "longest",
    "enable_batch_splitting": true
  }
}
```

### Network Optimization

#### Connection Pooling
```json
{
  "networking": {
    "keep_alive_timeout": 60,
    "max_connections": 1000,
    "connection_pool_size": 100,
    "enable_http2": true,
    "enable_compression": true
  }
}
```

## Monitoring and Observability

### Prometheus Metrics

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'openinferencev2'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: /metrics
    scrape_interval: 5s
```

### Key Metrics to Monitor

#### Performance Metrics
- `openinferencev2_requests_total`: Total number of requests
- `openinferencev2_request_duration_seconds`: Request latency histogram
- `openinferencev2_tokens_generated_total`: Total tokens generated
- `openinferencev2_batch_size`: Current batch size
- `openinferencev2_queue_length`: Request queue length

#### Resource Metrics
- `openinferencev2_gpu_utilization`: GPU utilization percentage
- `openinferencev2_gpu_memory_used`: GPU memory usage
- `openinferencev2_cpu_usage`: CPU usage percentage
- `openinferencev2_memory_usage`: System memory usage

#### Error Metrics
- `openinferencev2_errors_total`: Total number of errors
- `openinferencev2_timeouts_total`: Total number of timeouts
- `openinferencev2_failed_requests_total`: Failed requests

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "OpenInferencev2 Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(openinferencev2_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(openinferencev2_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P95 Latency"
          },
          {
            "expr": "histogram_quantile(0.99, rate(openinferencev2_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P99 Latency"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "openinferencev2_gpu_utilization",
            "legendFormat": "GPU {{gpu_id}}"
          }
        ]
      }
    ]
  }
}
```

## Security

### Authentication and Authorization

#### API Key Authentication
```json
{
  "security": {
    "authentication": {
      "type": "api_key",
      "header_name": "X-API-Key",
      "api_keys": [
        {
          "key": "your-secure-api-key",
          "name": "production-client",
          "permissions": ["inference", "monitoring"]
        }
      ]
    }
  }
}
```

#### TLS Configuration
```json
{
  "tls": {
    "enabled": true,
    "cert_file": "/etc/ssl/certs/openinferencev2.pem",
    "key_file": "/etc/ssl/private/openinferencev2.key",
    "protocols": ["TLSv1.2", "TLSv1.3"],
    "cipher_suites": ["ECDHE-ECDSA-AES256-GCM-SHA384"]
  }
}
```

### Rate Limiting

```json
{
  "rate_limiting": {
    "enabled": true,
    "requests_per_minute": 100,
    "burst_size": 20,
    "storage": "redis",
    "redis_url": "redis://localhost:6379"
  }
}
```

### Input Validation

```json
{
  "validation": {
    "max_prompt_length": 8192,
    "max_response_length": 4096,
    "allowed_models": ["llama2-7b", "llama2-13b"],
    "blocked_patterns": ["\\$\\{", "eval\\(", "<script"],
    "sanitize_input": true
  }
}
```

## Load Balancing

### NGINX Configuration

```nginx
upstream openinferencev2_backend {
    least_conn;
    server 10.0.1.10:8000 max_fails=3 fail_timeout=30s;
    server 10.0.1.11:8000 max_fails=3 fail_timeout=30s;
    server 10.0.1.12:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.openinferencev2.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.openinferencev2.com;
    
    ssl_certificate /etc/ssl/certs/openinferencev2.pem;
    ssl_certificate_key /etc/ssl/private/openinferencev2.key;
    
    location / {
        proxy_pass http://openinferencev2_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_timeout 60s;
        proxy_read_timeout 60s;
        proxy_send_timeout 60s;
        
        proxy_buffering off;
        proxy_cache off;
    }
    
    location /health {
        access_log off;
        proxy_pass http://openinferencev2_backend/health;
    }
}
```

## Backup and Recovery

### Model Backup

```bash
#!/bin/bash
# backup-models.sh

BACKUP_DIR="/backup/models"
MODEL_DIR="/opt/models"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
tar -czf "${BACKUP_DIR}/models_${DATE}.tar.gz" -C "${MODEL_DIR}" .

# Keep only last 7 days of backups
find "${BACKUP_DIR}" -name "models_*.tar.gz" -mtime +7 -delete

# Upload to S3 (optional)
aws s3 cp "${BACKUP_DIR}/models_${DATE}.tar.gz" s3://your-backup-bucket/models/
```

### Configuration Backup

```bash
#!/bin/bash
# backup-config.sh

BACKUP_DIR="/backup/config"
CONFIG_DIR="/etc/openinferencev2"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup configuration
cp -r "${CONFIG_DIR}" "${BACKUP_DIR}/config_${DATE}"

# Backup systemd service
cp /etc/systemd/system/openinferencev2.service "${BACKUP_DIR}/config_${DATE}/"
```

## Troubleshooting

### Common Issues

#### Out of Memory Errors
```bash
# Check GPU memory usage
nvidia-smi

# Reduce batch size in config
{
  "max_batch_size": 16,
  "kv_cache_size_gb": 8.0
}
```

#### High Latency
```bash
# Enable CUDA graphs
{
  "use_cuda_graphs": true,
  "enable_dynamic_batching": true,
  "max_wait_time": 0.05
}
```

#### Connection Issues
```bash
# Check service status
systemctl status openinferencev2

# Check logs
journalctl -u openinferencev2 -f

# Test connectivity
curl -f http://localhost:8000/health
```

### Performance Diagnostics

```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi

# Profile application
python -m cProfile -o profile.stats -m src.cli.main --model /models/llama2-7b

# Memory profiling
python -m memory_profiler -m src.cli.main --model /models/llama2-7b
```

### Log Analysis

```bash
# Search for errors
grep -i error /var/log/openinferencev2/openinferencev2.log

# Monitor real-time logs
tail -f /var/log/openinferencev2/openinferencev2.log

# Analyze performance
grep "inference_time" /var/log/openinferencev2/openinferencev2.log | awk '{print $NF}' | sort -n
```

## Scaling

### Horizontal Scaling

#### Multi-Node Setup
```yaml
# Scale up deployment
kubectl scale deployment openinferencev2 --replicas=5

# Add more nodes
kubectl taint nodes node1 node2 node3 nvidia.com/gpu=present:NoSchedule
```

#### Auto-scaling
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: openinferencev2-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: openinferencev2
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: openinferencev2_queue_length
      target:
        type: AverageValue
        averageValue: "10"
```

### Vertical Scaling

#### GPU Scaling
```json
{
  "engine": {
    "num_gpus": 4,
    "tensor_parallel_size": 4,
    "pipeline_parallel_size": 1
  }
}
```

## Production Checklist

### Pre-deployment
- [ ] Hardware requirements verified
- [ ] CUDA drivers installed and tested
- [ ] Model files downloaded and validated
- [ ] Configuration files reviewed
- [ ] Security settings configured
- [ ] Monitoring setup complete
- [ ] Backup procedures in place

### Deployment
- [ ] Service deployed successfully
- [ ] Health checks passing
- [ ] Load balancer configured
- [ ] SSL certificates installed
- [ ] Monitoring alerts configured
- [ ] Performance benchmarks completed

### Post-deployment
- [ ] Service monitoring active
- [ ] Performance metrics collected
- [ ] Error rates within acceptable limits
- [ ] Backup procedures tested
- [ ] Disaster recovery plan verified
- [ ] Documentation updated

## Support and Maintenance

### Regular Maintenance Tasks

#### Daily
- Monitor service health and performance
- Check error logs for issues
- Verify backup completion

#### Weekly
- Review performance metrics
- Update monitoring dashboards
- Rotate log files

#### Monthly
- Security patches and updates
- Model updates if available
- Performance optimization review

### Emergency Procedures

#### Service Outage
1. Check service status: `systemctl status openinferencev2`
2. Review logs: `journalctl -u openinferencev2 -n 100`
3. Restart service: `systemctl restart openinferencev2`
4. Verify health: `curl http://localhost:8000/health`

#### High Memory Usage
1. Check GPU memory: `nvidia-smi`
2. Reduce batch size temporarily
3. Restart service to clear memory
4. Investigate memory leaks

#### Performance Degradation
1. Check system resources
2. Review configuration settings
3. Analyze request patterns
4. Scale resources if needed

For additional support, please contact: nikjois@llamasearch.ai 