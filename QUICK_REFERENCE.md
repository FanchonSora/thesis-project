# 🧠 Brain Tumor Analysis - Quick Reference Card

## ⚡ Start Server (30 seconds)

### Windows
```powershell
cd c:\Source\Thesis\thesis-project
.\start_server.ps1
```

### Linux/Mac
```bash
cd ~/thesis-project
./start_server.sh
```

## 🌐 Access Points

| Purpose | URL |
|-------|-----|
| **Web UI** | http://localhost:8000 |
| **API Docs** | http://localhost:8000/docs |
| **Health Check** | http://localhost:8000/api/health |

---

## 📤 Upload MRI Files

**Required**: At least 3 out of 4 modalities
- ✅ T1 (Structural)
- ✅ T1ce (Contrast-enhanced)
- ✅ T2 (Fast spin echo)
- ✅ FLAIR (Fluid-suppressed)

**Format**: `.nii` or `.nii.gz`  
**Max Size**: 500 MB per file  
**Drag & Drop**: Supported!

---

## 📊 View Results

After processing, access:
- **Metrics Tab** - Tumor volumes & percentages
- **Details Tab** - Segmentation info
- **Volumes Tab** - 3D visualization

---

## 📥 Download

- 📄 **Report** - JSON with analysis results
- 🧬 **Prediction** - Segmentation masks (NIfTI)
- 🎨 **Mesh** - 3D file (OBJ/STL)

---

## 🔧 API Quick Reference

### Create Job
```bash
curl -X POST http://localhost:8000/jobs \
  -F "case_id=PATIENT001" \
  -F "t1=@t1.nii.gz" \
  -F "t1ce=@t1ce.nii.gz" \
  -F "t2=@t2.nii.gz" \
  -F "flair=@flair.nii.gz"
```

### Check Status
```bash
curl http://localhost:8000/jobs/JOB_ID_HERE
```

### Download Result
```bash
curl -O http://localhost:8000/jobs/JOB_ID_HERE/file/report
curl -O http://localhost:8000/jobs/JOB_ID_HERE/file/prediction
curl -O http://localhost:8000/jobs/JOB_ID_HERE/file/mesh
```

---

## 🛑 Stop Server

Press **Ctrl+C** in the terminal

---

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| **Port in use** | `.\start_server.ps1 -Port 8080` |
| **Dependencies missing** | `pip install -r models/synthesis-module/requirements.txt` |
| **Models not found** | Check model weight paths (see GETTING_STARTED.md) |
| **Slow processing** | Verify GPU support: `python -c "import torch; print(torch.cuda.is_available())"` |

---

## 📚 Documentation

| Guide | Purpose |
|-------|---------|
| **GETTING_STARTED.md** | Setup & first use |
| **SERVER_GUIDE.md** | Deployment & config |
| **DEVELOPER_GUIDE.md** | API integration |
| **PROJECT_COMPLETION.md** | What's been built |

---

## ✅ Pre-Launch Checklist

- [ ] Python 3.9+ installed
- [ ] Virtual environment activated
- [ ] FastAPI & dependencies installed
- [ ] Model files exist (check start_server output)
- [ ] Can read/write to `src/web_data/uploads`

---

## 🎯 Typical Workflow

1. **Start** → `.\start_server.ps1` (30 sec)
2. **Open** → http://localhost:8000 (5 sec)
3. **Upload** → Drag 4 MRI files + enter case ID (1 min)
4. **Submit** → Click "Analyze Brain" (10 sec)
5. **Wait** → Real-time progress tracking (2-10 min)
6. **View** → Results in tabs + 3D viewer (1 min)
7. **Download** → Reports, masks, meshes (30 sec)

**Total**: ~15-20 minutes per case

---

## 🚀 Quick Settings

### Development (Fast Reload)
```bash
.\start_server.ps1
```
*Code changes reload automatically*

### Production (Multiple Workers)
```powershell
.\start_server.ps1 -Workers 4
```
*Better concurrency, no reload*

### Custom Port
```powershell
.\start_server.ps1 -Port 8080
```

---

## 🔌 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.9 | 3.10+ |
| **RAM** | 8 GB | 16+ GB |
| **GPU** | Optional | CUDA 11.8+ |
| **Disk** | 10 GB | 30+ GB |
| **Browser** | Modern | Chrome/Firefox |

---

## 📞 Support

1. **Server won't start?** → Check `start_server.ps1` output
2. **Slow processing?** → Check for GPU support
3. **Upload fails?** → Verify NIfTI format and file size
4. **Results missing?** → Check `/jobs/{id}` status endpoint

See **GETTING_STARTED.md** for detailed troubleshooting.

---

## 💾 Where It's Stored

```
Input Files:     src/web_data/uploads/
Output Files:    src/web_output/
Logs:            logs/
```

---

## 🎉 You're Ready!

```powershell
cd c:\Source\Thesis\thesis-project
.\start_server.ps1
```

Then visit: **http://localhost:8000**

---

*Last Updated: January 2024*  
*Status: ✅ Production-Ready*
