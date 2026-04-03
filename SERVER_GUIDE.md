# Brain Tumor Analysis Server — Complete Setup Guide

**Production-ready web platform for MRI brain tumor analysis with 3D visualization**

---

## 🚀 Quick Start (2 Minutes)

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (GPU recommended)
- 8GB+ RAM, 50GB disk space
- All dependencies installed

### Start Server

```bash
# From project root
cd c:\Source\Thesis\thesis-project

# Activate environment
.\.venv\Scripts\Activate.ps1

# Start server
python -m uvicorn src.web_api:app --host 0.0.0.0 --port 8000 --reload

# Open browser
# Navigate to: http://localhost:8000
```

**Server running**: ✅ Access at `http://localhost:8000`

---

## 📋 Complete Setup Guide

### Phase 1: Environment Preparation

#### 1.1 Install Core Dependencies

```bash
# Navigate to project
cd c:\Source\Thesis\thesis-project

# Activate venv
.\.venv\Scripts\Activate.ps1

# Install FastAPI & web dependencies
pip install -q \
  fastapi==0.104.1 \
  uvicorn==0.24.0 \
  python-multipart==0.0.6 \
  python-dotenv==1.0.0

# Verify installation
python -c "import fastapi; print('FastAPI OK')"
```

#### 1.2 Prepare Model Weights

```bash
# Check model files exist
ls -lh src/segmentation-module/model-weight/
ls -lh src/synthesis-module/model-weight/

# Expected output:
#   final_model_unet.pth       (~460 MB)
#   epoch_118.pth              (~500 MB)
```

If models don't exist, download them from:
- Segmentation: [Link to model](http://example.com/final_model_unet.pth)
- Synthesis: [Link to model](http://example.com/epoch_118.pth)

#### 1.3 Prepare Data Directories

```bash
# Create necessary directories
mkdir -p src/web_data/uploads
mkdir -p src/web_output
mkdir -p logs

# Verify
ls -ld src/web_data/uploads src/web_output logs
```

---

### Phase 2: Server Configuration

#### 2.1 Environment Variables (Optional)

Create `.env` file in project root:

```bash
# .env
ENVIRONMENT=production
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
MAX_UPLOAD_SIZE=500000000  # 500MB
TEMP_DIR=./src/web_data/uploads
OUTPUT_DIR=./src/web_output
LOG_DIR=./logs
GPU_ID=0
DEVICE=cuda
MODEL_PRECISION=float32
```

#### 2.2 Server Configuration

Edit if needed:

```python
# src/web_api.py
DATA_ROOT = APP_ROOT / "web_data" / "uploads"  # Temp storage
OUTPUT_ROOT = APP_ROOT / "web_output"           # Results storage
MAX_UPLOAD_MB = 500                             # File size limit
```

---

### Phase 3: Start Server

#### 3.1 Development Mode (Single Worker, Auto-reload)

```bash
# Perfect for testing & development
python -m uvicorn src.web_api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --log-level info
```

**Output**:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

#### 3.2 Production Mode (Multiple Workers)

```bash
# For production deployment
python -m uvicorn src.web_api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level warning
```

#### 3.3 With Custom Configuration

```bash
# Custom host & port
python -m uvicorn src.web_api:app \
  --host 192.168.1.100 \
  --port 5000 \
  --workers 2

# Access at: http://192.168.1.100:5000
```

#### 3.4 Run in Background (PowerShell)

```powershell
# Start in background
$job = Start-Job -ScriptBlock {
    cd c:\Source\Thesis\thesis-project
    .\.venv\Scripts\Activate.ps1
    python -m uvicorn src.web_api:app --host 0.0.0.0 --port 8000
}

# Check status
Get-Job $job
Receive-Job $job

# Stop server
Stop-Job $job
Remove-Job $job
```

#### 3.5 Run with Gunicorn (Linux/macOS)

```bash
pip install gunicorn

gunicorn \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  src.web_api:app
```

---

## 🌐 Accessing the Platform

### Web Interface

Open in web browser:

| URL | Purpose |
|-----|---------|
| `http://localhost:8000` | Main UI |
| `http://localhost:8000/api/health` | Health check |
| `http://localhost:8000/docs` | API documentation |
| `http://localhost:8000/redoc` | API docs (ReDoc) |

### API Endpoints

#### Upload & Analysis

```bash
# Submit analysis (multipart form data)
curl -X POST http://localhost:8000/jobs \
  -F case_id=BraTS2021_00000 \
  -F t1=@brain_t1.nii.gz \
  -F t1ce=@brain_t1ce.nii.gz \
  -F t2=@brain_t2.nii.gz \
  -F flair=@brain_flair.nii.gz

# Response:
# {"job_id": "uuid-here", "status": "queued"}
```

#### Check Job Status

```bash
# Get job status
curl http://localhost:8000/jobs/{job_id}

# Response:
# {"job_id": "...", "case_id": "...", "status": "running", ...}
```

#### Download Results

```bash
# Get report (JSON)
curl http://localhost:8000/jobs/{job_id}/report > report.json

# Get prediction (NIfTI)
curl http://localhost:8000/jobs/{job_id}/file/pred_post > prediction.nii.gz

# Get mesh (OBJ)
curl http://localhost:8000/jobs/{job_id}/file/mesh > mesh.obj
```

---

## 🔧 Troubleshooting

### Issue: Port Already in Use

```bash
# Find process using port 8000
netstat -ano | findstr :8000  # Windows
lsof -i :8000                  # macOS/Linux

# Kill process (Windows)
taskkill /PID {PID} /F

# Use different port
python -m uvicorn src.web_api:app --port 8001
```

### Issue: CUDA Out of Memory

```bash
# Use CPU inference
# Edit src/web_api.py and change:
# device="cuda" → device="cpu"

# Or reduce batch size in pipeline processing
```

### Issue: Slow File Upload

Server defaults to 500MB max upload. Files are processed in background:
- Frontend shows progress
- No timeout on long analysis
- Results ready when complete

### Issue: 404 on Static Files

```bash
# Verify file structure
ls src/web_data/
# Should show: index.html, style.css, app.js

# Clear browser cache (Ctrl+Shift+Delete)
# or use incognito mode
```

### Issue: API Returns 500 Error

```bash
# Check server logs for detailed error:
# Look at terminal output
# Or check logs/:
tail -f logs/error.log

# Common issues:
# - Model weights file not found
# - CUDA out of memory
# - Invalid input file format
# - Disk full
```

---

## 📊 Monitoring & Performance

### Check Server Health

```bash
# Curl to health endpoint
curl http://localhost:8000/api/health
# {"status": "ok"}

# Monitor job queue
curl http://localhost:8000/api/stats
# Shows active jobs, queue length, etc.
```

### Performance Tips

1. **GPU Optimization**
   ```bash
   # Enable TF32 for faster training
   export CUDA_LAUNCH_BLOCKING=0
   export CUBLAS_WORKSPACE_CONFIG=:16:8
   ```

2. **Multi-GPU (if available)**
   ```python
   # In web_api.py, use distributed inference
   device = "cuda:0"  # or cuda:1, cuda:2, etc.
   ```

3. **Memory Management**
   - Reduce `MAX_UPLOAD_MB` if needed
   - Clean up old uploads: `rm -rf src/web_data/uploads/*`
   - Monitor disk: `df -h`

4. **Load Balancing** (Multiple Servers)
   ```bash
   # Start 3 instances on different ports
   for port in 8000 8001 8002; do
     python -m uvicorn src.web_api:app --port $port &
   done
   
   # Use nginx or similar as reverse proxy
   ```

---

## 🔐 Security Considerations

### For Production Deployment

1. **Enable HTTPS**
   ```bash
   pip install python-multipart
   
   # Use reverse proxy (nginx, Apache, etc.)
   # Configure SSL certificates
   ```

2. **Restrict Upload Size**
   ```python
   # In web_api.py
   MAX_UPLOAD_MB = 500  # Adjust as needed
   ```

3. **API Authentication** (Optional)
   ```python
   from fastapi.security import HTTPBearer
   
   security = HTTPBearer()
   
   @app.post("/jobs")
   async def create_job(..., credentials: HTTPAuthCredentials = Depends(security)):
       # Verify token
       pass
   ```

4. **Rate Limiting**
   ```bash
   pip install slowapi
   
   # Use SlowAPI middleware to rate limit requests
   ```

5. **CORS Configuration**
   ```python
   # In web_api.py - restrict origins
   allow_origins=["https://yourdomain.com"]  # Instead of ["*"]
   ```

---

## 📦 Docker Deployment (Optional)

### Dockerfile

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

COPY . .
RUN pip install -r requirements.txt
RUN pip install fastapi uvicorn

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.web_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./src/web_data:/app/src/web_data
      - ./src/web_output:/app/src/web_output
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - DEVICE=cuda
    runtime: nvidia
```

### Build & Run

```bash
# Build image
docker build -t brain-analysis:latest .

# Run container
docker run -p 8000:8000 --gpus all brain-analysis:latest

# With docker-compose
docker-compose up -d
```

---

## 📈 Expected Performance

### Analysis Time per Case

| Component | Time | GPU | CPU |
|-----------|------|-----|-----|
| Upload | <5s | <5s | <5s |
| Preprocessing | 2-5s | 2-5s | 5-10s |
| Synthesis (if missing) | 10-30s | 10-30s | 2-5m |
| Segmentation | 5-15s | 5-15s | 30-60s |
| Visualization | 2-5s | 2-5s | 2-5s |
| **Total** | **30-60s** | **30-60s** | **3-10m** |

### Server Metrics

| Metric | Value |
|--------|-------|
| Max concurrent jobs | 4 (default) |
| Max upload size | 500 MB |
| Request timeout | 30 minutes |
| Storage per case | ~100-200 MB |

---

## 📝 Logging

### View Logs

```bash
# Real-time logs
tail -f logs/server.log

# Filter errors only
grep ERROR logs/server.log

# Check specific date
grep "2024-04-01" logs/server.log
```

### Log Levels

- `DEBUG`: Detailed information
- `INFO`: General information
- `WARNING`: Warnings
- `ERROR`: Errors
- `CRITICAL`: Critical issues

### Configure Logging

```python
# In web_api.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/server.log'),
        logging.StreamHandler()
    ]
)
```

---

## 🛑 Shutdown

### Graceful Shutdown

```bash
# Ctrl+C in terminal where server is running
# Server will finish current jobs before stopping

# Or via process management
kill -SIGTERM {pid}
```

### Clean Up

```bash
# Remove old uploads
rm -rf src/web_data/uploads/*

# Clear outputs (keep recent 10)
ls -1tr src/web_output | head -n -10 | xargs -I {} rm -rf src/web_output/{}

# Check disk usage
du -sh src/web_data src/web_output
```

---

## 🎯 Next Steps

1. **Check Health**: `curl http://localhost:8000/api/health`
2. **Upload Test Case**: Use web UI
3. **Review Results**: Check 3D visualization
4. **Download Report**: Export JSON analysis
5. **Deploy to Production**: Follow Docker setup

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Guide](https://www.uvicorn.org/)
- [Brain Tumor Analysis README](../../README.md)
- [Synthesis Module Guide](../../models/diffusion-for-mri-tumor-brain-creation/SYNTHESIS_QUICK_START.md)

---

**Server Setup Completed** ✅  
**Ready for production deployment**
