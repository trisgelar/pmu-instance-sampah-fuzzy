# Deployment Guide

Complete deployment guide for the Waste Detection System with Fuzzy Classification.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Google Colab Deployment](#google-colab-deployment)
- [Local Development Deployment](#local-development-deployment)
- [Production Deployment](#production-deployment)
- [Edge Device Deployment](#edge-device-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Troubleshooting](#troubleshooting)

## Overview

This guide covers deployment scenarios for the waste detection system across different environments:

- **Google Colab**: Cloud-based development and training
- **Local Development**: Local machine setup for development
- **Production**: High-performance production deployment
- **Edge Devices**: RKNN-based edge deployment
- **Cloud Platforms**: AWS, GCP, Azure deployment

## Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: 4 cores, 2.0 GHz
- **RAM**: 8 GB
- **Storage**: 20 GB free space
- **GPU**: NVIDIA GPU with 4GB VRAM (recommended)
- **OS**: Linux, Windows, or macOS

#### Recommended Requirements
- **CPU**: 8+ cores, 3.0 GHz
- **RAM**: 16+ GB
- **Storage**: 50+ GB SSD
- **GPU**: NVIDIA RTX 3060 or better
- **OS**: Ubuntu 20.04+ or Windows 10+

### Software Requirements

#### Python Environment
```bash
# Python 3.8 or higher
python --version  # Should be 3.8+

# Virtual environment (recommended)
python -m venv waste_detection_env
source waste_detection_env/bin/activate  # Linux/Mac
# or
waste_detection_env\Scripts\activate  # Windows
```

#### Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import ultralytics; print('Ultralytics installed successfully')"
python -c "import skfuzzy; print('Scikit-fuzzy installed successfully')"
```

### API Keys and Credentials

#### Required
- **Roboflow API Key**: For dataset access
- **Google Drive Access**: For file storage (optional)

#### Optional
- **Google Cloud API Key**: For cloud deployment
- **AWS Credentials**: For AWS deployment
- **Azure Credentials**: For Azure deployment

## Google Colab Deployment

### 1. Setup Google Colab

#### Create New Notebook
1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Enable GPU acceleration:
   - Runtime → Change runtime type
   - Hardware accelerator: GPU

#### Upload Project Files
```python
# Upload project files to Colab
from google.colab import files
import zipfile
import os

# Upload your project zip file
uploaded = files.upload()

# Extract files
with zipfile.ZipFile('your_project.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

# Verify files
!ls -la
```

### 2. Install Dependencies

```python
# Install required packages
!pip install ultralytics roboflow opencv-python pyyaml matplotlib pandas numpy scikit-fuzzy

# Install additional packages if needed
!pip install google-drive-api  # For Google Drive integration
```

### 3. Configure Secrets

```python
# Create secrets.yaml
secrets_content = """
roboflow_api_key: "YOUR_ROBOFLOW_API_KEY_HERE"
"""

with open('secrets.yaml', 'w') as f:
    f.write(secrets_content)

# Verify secrets
!python validate_secrets.py
```

### 4. Run Training Pipeline

```python
# Import the main system
from main_colab import WasteDetectionSystemColab

# Initialize system
system = WasteDetectionSystemColab()

# Train model
result = system.train_and_export_model("v8n", epochs=50, batch_size=16)

# Run inference
inference_results = system.run_inference_and_visualization(result, "v8n")

# Analyze results
metrics = system.analyze_training_run(result, "v8n")
```

### 5. Save Results to Google Drive

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy results to Drive
!cp -r runs /content/drive/MyDrive/waste_detection_results/
```

### Colab Best Practices

#### Memory Management
```python
# Monitor GPU memory
!nvidia-smi

# Clear GPU cache if needed
import torch
torch.cuda.empty_cache()
```

#### Long Training Sessions
```python
# Prevent Colab disconnection
from google.colab import runtime
runtime.raise_for_unexpected_disconnects()

# Save checkpoints regularly
# The system automatically saves checkpoints every 10 epochs
```

#### Error Handling
```python
try:
    result = system.train_and_export_model("v8n")
except Exception as e:
    print(f"Training failed: {e}")
    # Save any partial results
    system.save_partial_results()
```

## Local Development Deployment

### 1. Environment Setup

#### Clone Repository
```bash
git clone <repository-url>
cd pmu-instance-sampah-fuzzy
```

#### Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

#### Configure Environment
```bash
# Create secrets file
cp secrets.yaml.example secrets.yaml
# Edit secrets.yaml with your API keys

# Create config file (if not exists)
# The system will create default config.yaml if not present
```

### 2. Development Workflow

#### Run Tests
```bash
# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --category security
python run_tests.py --category config
```

#### Development Mode
```python
# Initialize system in development mode
system = WasteDetectionSystemColab(environment="development")

# Use smaller datasets for faster iteration
system.update_configuration("dataset", "roboflow_version", "1")
system.update_configuration("model", "default_epochs", 5)
```

#### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug information
system = WasteDetectionSystemColab()
system.train_and_export_model("v8n", epochs=2)  # Quick test
```

### 3. Local GPU Setup

#### NVIDIA GPU Setup
```bash
# Install CUDA toolkit
# Follow NVIDIA's official guide for your OS

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### AMD GPU Setup
```bash
# Install ROCm (Linux only)
# Follow AMD's official guide

# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

### 4. Local Testing

#### Unit Tests
```bash
# Run all tests
python run_tests.py

# Run with coverage
pip install coverage
coverage run run_tests.py
coverage report
```

#### Integration Tests
```python
# Test complete workflow
system = WasteDetectionSystemColab()

# Test training
result = system.train_and_export_model("v8n", epochs=2)

# Test inference
inference_results = system.run_inference_and_visualization(result, "v8n")

# Test analysis
metrics = system.analyze_training_run(result, "v8n")
```

## Production Deployment

### 1. Production Environment Setup

#### Server Requirements
- **CPU**: 16+ cores
- **RAM**: 32+ GB
- **GPU**: NVIDIA RTX 4090 or better
- **Storage**: 1TB+ NVMe SSD
- **Network**: High-speed internet connection

#### Operating System
```bash
# Ubuntu 20.04 LTS or higher
sudo apt update
sudo apt upgrade

# Install essential packages
sudo apt install python3 python3-pip python3-venv git curl wget
```

### 2. Production Configuration

#### Environment Variables
```bash
# Set production environment variables
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export MODEL_CACHE_DIR=/opt/waste_detection/models
export DATA_DIR=/opt/waste_detection/data
export RESULTS_DIR=/opt/waste_detection/results
```

#### Production Config
```yaml
# config_production.yaml
model:
  default_epochs: 100
  default_batch_size: 32
  default_img_size: [832, 832]
  default_conf_threshold: 0.3

system:
  environment: production
  num_workers: 8
  pin_memory: true
  mixed_precision: true
  request_timeout: 600

logging:
  level: INFO
  file_logging: true
  console_logging: false
  log_file: /var/log/waste_detection.log
```

### 3. Production Deployment Steps

#### 1. Install Dependencies
```bash
# Create production environment
python3 -m venv /opt/waste_detection/venv
source /opt/waste_detection/venv/bin/activate

# Install production dependencies
pip install -r requirements.txt

# Install additional production packages
pip install gunicorn uvicorn fastapi
```

#### 2. Setup Directory Structure
```bash
# Create production directories
sudo mkdir -p /opt/waste_detection/{models,data,results,logs}
sudo chown -R $USER:$USER /opt/waste_detection
```

#### 3. Configure Secrets
```bash
# Create production secrets
sudo nano /opt/waste_detection/secrets.yaml

# Set proper permissions
sudo chmod 600 /opt/waste_detection/secrets.yaml
```

#### 4. Setup Systemd Service
```bash
# Create systemd service file
sudo nano /etc/systemd/system/waste-detection.service

[Unit]
Description=Waste Detection System
After=network.target

[Service]
Type=simple
User=waste_detection
WorkingDirectory=/opt/waste_detection
Environment=PATH=/opt/waste_detection/venv/bin
ExecStart=/opt/waste_detection/venv/bin/python main_colab.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 5. Start Service
```bash
# Enable and start service
sudo systemctl enable waste-detection
sudo systemctl start waste-detection

# Check status
sudo systemctl status waste-detection
```

### 4. Production Monitoring

#### Log Monitoring
```bash
# Monitor logs
sudo journalctl -u waste-detection -f

# Check log files
tail -f /var/log/waste_detection.log
```

#### Performance Monitoring
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop
```

#### Health Checks
```python
# Health check script
import requests
import json

def health_check():
    try:
        response = requests.get('http://localhost:8000/health')
        return response.status_code == 200
    except:
        return False

if health_check():
    print("System is healthy")
else:
    print("System is unhealthy")
```

## Edge Device Deployment

### 1. RKNN Model Conversion

#### Prerequisites
```bash
# Install RKNN Toolkit
# Download from Rockchip official website
# Follow installation guide for your platform

# Verify installation
python -c "import rknn_toolkit; print('RKNN Toolkit installed')"
```

#### Convert Model
```python
from modules.rknn_converter import RknnConverter

# Convert ONNX to RKNN
converter = RknnConverter("models/best.onnx")
rknn_path = converter.convert_to_rknn(target_platform="rk3588")

# Validate conversion
success = converter.validate_rknn_model(rknn_path)
print(f"Conversion successful: {success}")
```

### 2. Edge Device Setup

#### RK3588 Setup
```bash
# Install RKNN runtime
sudo apt update
sudo apt install rknn-toolkit2

# Verify installation
python -c "import rknn_toolkit2; print('RKNN Runtime installed')"
```

#### Edge Application
```python
# Edge inference application
import cv2
import numpy as np
from rknn.api import RKNN

class EdgeWasteDetector:
    def __init__(self, rknn_path):
        self.rknn = RKNN(verbose=True)
        self.rknn.load_rknn(rknn_path)
        self.rknn.init_runtime()
    
    def detect_waste(self, image):
        # Preprocess image
        processed = self.preprocess_image(image)
        
        # Run inference
        outputs = self.rknn.inference(inputs=[processed])
        
        # Postprocess results
        results = self.postprocess_outputs(outputs)
        
        return results
    
    def preprocess_image(self, image):
        # Resize to model input size
        resized = cv2.resize(image, (640, 640))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def postprocess_outputs(self, outputs):
        # Process model outputs
        # Convert to bounding boxes and classifications
        return outputs

# Usage
detector = EdgeWasteDetector("models/waste_detection.rknn")
results = detector.detect_waste(image)
```

### 3. Edge Device Optimization

#### Memory Optimization
```python
# Optimize for edge devices
class OptimizedEdgeDetector:
    def __init__(self, rknn_path):
        self.rknn = RKNN(verbose=False)  # Reduce verbosity
        self.rknn.load_rknn(rknn_path)
        self.rknn.init_runtime(target='rk3588')
    
    def detect_batch(self, images, batch_size=4):
        # Process images in batches
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_results = self.process_batch(batch)
            results.extend(batch_results)
        return results
```

#### Power Optimization
```python
# Power-efficient inference
class PowerOptimizedDetector:
    def __init__(self, rknn_path):
        self.rknn = RKNN(verbose=False)
        self.rknn.load_rknn(rknn_path)
        
        # Use power-efficient runtime
        self.rknn.init_runtime(target='rk3588', 
                              device_id='0',
                              perf_debug=False)
    
    def detect_with_power_management(self, image):
        # Implement power management
        # Reduce inference frequency
        # Use lower precision when possible
        pass
```

## Cloud Deployment

### 1. AWS Deployment

#### AWS Setup
```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure
```

#### AWS Infrastructure
```yaml
# infrastructure/aws/main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

# EC2 instance for training
resource "aws_instance" "training_instance" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "g4dn.xlarge"
  
  tags = {
    Name = "waste-detection-training"
  }
}

# S3 bucket for model storage
resource "aws_s3_bucket" "model_bucket" {
  bucket = "waste-detection-models"
}
```

#### AWS Deployment Script
```bash
#!/bin/bash
# deploy_aws.sh

# Build Docker image
docker build -t waste-detection .

# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
docker tag waste-detection:latest $AWS_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/waste-detection:latest
docker push $AWS_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/waste-detection:latest

# Deploy to ECS
aws ecs update-service --cluster waste-detection-cluster --service waste-detection-service --force-new-deployment
```

### 2. Google Cloud Deployment

#### GCP Setup
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize GCP
gcloud init
```

#### GCP Infrastructure
```yaml
# infrastructure/gcp/main.tf
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = "waste-detection-project"
  region  = "us-central1"
}

# Compute Engine instance
resource "google_compute_instance" "training_instance" {
  name         = "waste-detection-training"
  machine_type = "n1-standard-4"
  zone         = "us-central1-a"
  
  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-10"
    }
  }
}
```

#### GCP Deployment Script
```bash
#!/bin/bash
# deploy_gcp.sh

# Build and push to Container Registry
docker build -t gcr.io/$PROJECT_ID/waste-detection .
docker push gcr.io/$PROJECT_ID/waste-detection

# Deploy to Cloud Run
gcloud run deploy waste-detection \
  --image gcr.io/$PROJECT_ID/waste-detection \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### 3. Azure Deployment

#### Azure Setup
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login
```

#### Azure Infrastructure
```yaml
# infrastructure/azure/main.tf
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# Azure Container Registry
resource "azurerm_container_registry" "acr" {
  name                = "wasteDetectionRegistry"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Standard"
}
```

## Monitoring & Maintenance

### 1. System Monitoring

#### Health Checks
```python
# health_check.py
import requests
import psutil
import time

class SystemMonitor:
    def __init__(self):
        self.endpoints = [
            "http://localhost:8000/health",
            "http://localhost:8000/metrics"
        ]
    
    def check_system_health(self):
        health_status = {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "endpoints": {}
        }
        
        for endpoint in self.endpoints:
            try:
                response = requests.get(endpoint, timeout=5)
                health_status["endpoints"][endpoint] = response.status_code == 200
            except:
                health_status["endpoints"][endpoint] = False
        
        return health_status
    
    def log_health_status(self):
        status = self.check_system_health()
        print(f"Health Status: {status}")
        return status

# Usage
monitor = SystemMonitor()
monitor.log_health_status()
```

#### Performance Monitoring
```python
# performance_monitor.py
import time
import psutil
import logging

class PerformanceMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def monitor_training_performance(self, training_function):
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        try:
            result = training_function()
            
            end_time = time.time()
            end_memory = psutil.virtual_memory().used
            
            performance_metrics = {
                "execution_time": end_time - start_time,
                "memory_used": end_memory - start_memory,
                "success": True
            }
            
            self.logger.info(f"Training completed: {performance_metrics}")
            return result, performance_metrics
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return None, {"success": False, "error": str(e)}

# Usage
monitor = PerformanceMonitor()
result, metrics = monitor.monitor_training_performance(
    lambda: system.train_and_export_model("v8n")
)
```

### 2. Log Management

#### Logging Configuration
```python
# logging_config.py
import logging
import logging.handlers
import os

def setup_logging(log_level="INFO", log_file="/var/log/waste_detection.log"):
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Usage
logger = setup_logging("INFO", "/var/log/waste_detection.log")
logger.info("Waste detection system started")
```

### 3. Backup and Recovery

#### Backup Strategy
```bash
#!/bin/bash
# backup.sh

# Backup configuration
cp config.yaml backup/config_$(date +%Y%m%d_%H%M%S).yaml

# Backup models
tar -czf backup/models_$(date +%Y%m%d_%H%M%S).tar.gz runs/

# Backup logs
tar -czf backup/logs_$(date +%Y%m%d_%H%M%S).tar.gz logs/

# Upload to cloud storage
aws s3 sync backup/ s3://waste-detection-backups/
```

#### Recovery Procedures
```bash
#!/bin/bash
# recovery.sh

# Restore configuration
cp backup/config_$(date +%Y%m%d_%H%M%S).yaml config.yaml

# Restore models
tar -xzf backup/models_$(date +%Y%m%d_%H%M%S).tar.gz

# Restore logs
tar -xzf backup/logs_$(date +%Y%m%d_%H%M%S).tar.gz

# Restart services
sudo systemctl restart waste-detection
```

## Troubleshooting

### Common Issues

#### 1. GPU Memory Issues
```python
# Solution: Reduce batch size and image size
system.update_configuration("model", "default_batch_size", 8)
system.update_configuration("model", "default_img_size", [512, 512])

# Clear GPU cache
import torch
torch.cuda.empty_cache()
```

#### 2. API Key Errors
```bash
# Check secrets configuration
python validate_secrets.py

# Verify API key format
# Roboflow keys should be long alphanumeric strings
```

#### 3. Model Training Failures
```python
# Check dataset availability
from modules.dataset_manager import DatasetManager
dataset_manager = DatasetManager("your_project", "1")
dataset_manager.validate_dataset()

# Check model configuration
config = system.get_configuration_summary()
print(config)
```

#### 4. Performance Issues
```python
# Monitor system resources
import psutil
print(f"CPU Usage: {psutil.cpu_percent()}%")
print(f"Memory Usage: {psutil.virtual_memory().percent}%")

# Optimize configuration
system.update_configuration("system", "num_workers", 4)
system.update_configuration("system", "mixed_precision", True)
```

### Debug Procedures

#### 1. Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug information
system = WasteDetectionSystemColab()
system.train_and_export_model("v8n", epochs=1)
```

#### 2. Test Individual Components
```python
# Test configuration
from modules.config_manager import ConfigManager
config_manager = ConfigManager()
print(config_manager.get_config_summary())

# Test fuzzy classification
from modules.fuzzy_area_classifier import FuzzyAreaClassifier
classifier = FuzzyAreaClassifier()
result = classifier.classify_area(15.5)
print(f"Classification: {result}")
```

#### 3. Validate System Integrity
```bash
# Run all tests
python run_tests.py

# Check file permissions
ls -la secrets.yaml
ls -la config.yaml

# Verify dependencies
pip list | grep -E "(ultralytics|roboflow|skfuzzy)"
```

### Emergency Procedures

#### 1. System Recovery
```bash
# Stop all services
sudo systemctl stop waste-detection

# Clear temporary files
rm -rf /tmp/waste_detection_*

# Restart services
sudo systemctl start waste-detection
```

#### 2. Data Recovery
```bash
# Restore from backup
./recovery.sh

# Verify data integrity
python validate_secrets.py
python run_tests.py
```

#### 3. Emergency Contact
```bash
# Send emergency notification
curl -X POST -H "Content-Type: application/json" \
  -d '{"text":"Waste detection system emergency"}' \
  https://hooks.slack.com/services/YOUR_WEBHOOK_URL
```

---

This deployment guide provides comprehensive instructions for deploying the waste detection system across different environments. For additional support, refer to the troubleshooting documentation or contact the development team. 