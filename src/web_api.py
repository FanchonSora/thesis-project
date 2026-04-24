from __future__ import annotations
import json
import logging
import mimetypes
import shutil
import traceback
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional
import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .preprocessing import build_modality_paths
from .run_pipeline import process_case
logger = logging.getLogger("brain_api")
logging.basicConfig(level=logging.INFO)
APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent

FRONTEND_ROOT = APP_ROOT / "web_data"
DATA_ROOT = APP_ROOT / "web_data" / "uploads"
OUTPUT_ROOT = APP_ROOT / "web_output"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI(
    title="Brain Tumor Analysis API",
    description="Advanced MRI brain tumor segmentation platform",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(FRONTEND_ROOT)), name="static")
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = Lock()
PREVIEW_PLANES = ("axial", "coronal", "sagittal")
DEFAULT_MESH_LOD = "low"

def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj

def _json_response(content: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        content=to_jsonable(content),
        status_code=status_code,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

def _save_upload(upload: UploadFile, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh)

def _set_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        current = JOBS.get(job_id, {}).copy()
        current.update(updates)
        JOBS[job_id] = current

def _get_job_snapshot(job_id: str) -> Optional[Dict[str, Any]]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job is not None else None

def _list_jobs_snapshot() -> Dict[str, Dict[str, Any]]:
    with JOBS_LOCK:
        return {jid: dict(job) for jid, job in JOBS.items()}

def _get_case_dir(job_id: str) -> Path:
    return OUTPUT_ROOT / job_id

def _get_report_path(job_id: str) -> Path:
    return _get_case_dir(job_id) / f"{job_id}_report.json"

def _load_report_from_disk(job_id: str) -> Optional[Dict[str, Any]]:
    report_path = _get_report_path(job_id)
    if not report_path.exists():
        return None
    try:
        with report_path.open("r", encoding="utf-8") as fh:
            return to_jsonable(json.load(fh))
    except Exception:
        logger.error("[JOB %s] Failed reading report from disk:\n%s", job_id, traceback.format_exc())
        return None

def _ensure_job_report_loaded(job_id: str) -> Optional[Dict[str, Any]]:
    job = _get_job_snapshot(job_id)
    if not job:
        return None
    if isinstance(job.get("report"), dict):
        return job["report"]
    report = _load_report_from_disk(job_id)
    if report is not None:
        _set_job(job_id, report=report)
    return report

def _safe_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None

def _file_size(path_str: Optional[str]) -> Optional[int]:
    path = _safe_path(path_str)
    return path.stat().st_size if path else None

def _build_status_payload(job_id: str, job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "case_id": job.get("case_id"),
        "status": job.get("status", "queued"),
        "error": job.get("error"),
        "has_report": isinstance(job.get("report"), dict) or _get_report_path(job_id).exists(),
    }

def _summary_from_report(job_id: str, report: Dict[str, Any]) -> Dict[str, Any]:
    paths = report.get("paths", {}) if isinstance(report, dict) else {}
    preview_paths = paths.get("preview_paths", {}) or {}
    mesh_paths = paths.get("mesh_paths", {}) or {}
    brain_mesh_paths = paths.get("brain_mesh_paths", {}) or {}
    wt_mesh_paths = paths.get("wt_mesh_paths", {}) or {}
    tc_mesh_paths = paths.get("tc_mesh_paths", {}) or {}
    et_mesh_paths = paths.get("et_mesh_paths", {}) or {}
    metadata = report.get("metadata", {}) or {}
    preprocess = metadata.get("preprocess", {}) or {}
    asset_sizes = {
        "prediction_bytes": _file_size(paths.get("pred_post_path")),
        "report_bytes": _file_size(paths.get("report_path")),
        "mesh_low_bytes": _file_size(mesh_paths.get("low") or paths.get("mesh_path")),
        "mesh_medium_bytes": _file_size(mesh_paths.get("medium")),
        "mesh_high_bytes": _file_size(mesh_paths.get("high")),
        "brain_mesh_low_bytes": _file_size(brain_mesh_paths.get("low") or paths.get("brain_mesh_path")),
        "brain_mesh_medium_bytes": _file_size(brain_mesh_paths.get("medium")),
        "brain_mesh_high_bytes": _file_size(brain_mesh_paths.get("high")),
        "wt_mesh_low_bytes": _file_size(wt_mesh_paths.get("low") or paths.get("wt_mesh_path")),
        "wt_mesh_medium_bytes": _file_size(wt_mesh_paths.get("medium")),
        "wt_mesh_high_bytes": _file_size(wt_mesh_paths.get("high")),
        "tc_mesh_low_bytes": _file_size(tc_mesh_paths.get("low") or paths.get("tc_mesh_path")),
        "tc_mesh_medium_bytes": _file_size(tc_mesh_paths.get("medium")),
        "tc_mesh_high_bytes": _file_size(tc_mesh_paths.get("high")),
        "et_mesh_low_bytes": _file_size(et_mesh_paths.get("low") or paths.get("et_mesh_path")),
        "et_mesh_medium_bytes": _file_size(et_mesh_paths.get("medium")),
        "et_mesh_high_bytes": _file_size(et_mesh_paths.get("high")),
    }
    return {
        "job_id": job_id,
        "case_id": report.get("case_id"),
        "status": report.get("status"),
        "synthesis_status": report.get("synthesis_status"),
        "downsample_factor": report.get("downsample_factor"),
        "missing_flags": report.get("missing_flags", {}),
        "region_volumes_voxels": report.get("region_volumes_voxels", {}),
        "region_volumes_mm3": report.get("region_volumes_mm3", {}),
        "preview": {
            plane: f"/jobs/{job_id}/preview/{plane}"
            for plane in PREVIEW_PLANES
            if preview_paths.get(plane)
        },
        "viewer": {
            "default_lod": DEFAULT_MESH_LOD,
            "available_lods": [
                lod for lod in ("low", "medium", "high")
                if any([brain_mesh_paths.get(lod), wt_mesh_paths.get(lod), tc_mesh_paths.get(lod), et_mesh_paths.get(lod)])
            ],
            "mesh_url": f"/jobs/{job_id}/file/brain_mesh?lod={DEFAULT_MESH_LOD}",
            "brain": {lod: f"/jobs/{job_id}/file/brain_mesh?lod={lod}" for lod in ("low", "medium", "high")},
            "regions": {
                "wt": {lod: f"/jobs/{job_id}/file/wt_mesh?lod={lod}" for lod in ("low", "medium", "high")},
                "tc": {lod: f"/jobs/{job_id}/file/tc_mesh?lod={lod}" for lod in ("low", "medium", "high")},
                "et": {lod: f"/jobs/{job_id}/file/et_mesh?lod={lod}" for lod in ("low", "medium", "high")},
            },
        },
        "downloads": {
            "report": f"/jobs/{job_id}/file/report",
            "prediction": f"/jobs/{job_id}/file/pred_post",
            "mesh_low": f"/jobs/{job_id}/file/mesh?lod=low",
            "mesh_medium": f"/jobs/{job_id}/file/mesh?lod=medium",
            "mesh_high": f"/jobs/{job_id}/file/mesh?lod=high",
            "brain_mesh_low": f"/jobs/{job_id}/file/brain_mesh?lod=low",
            "brain_mesh_medium": f"/jobs/{job_id}/file/brain_mesh?lod=medium",
            "brain_mesh_high": f"/jobs/{job_id}/file/brain_mesh?lod=high",
            "wt_mesh_low": f"/jobs/{job_id}/file/wt_mesh?lod=low",
            "wt_mesh_medium": f"/jobs/{job_id}/file/wt_mesh?lod=medium",
            "wt_mesh_high": f"/jobs/{job_id}/file/wt_mesh?lod=high",
            "tc_mesh_low": f"/jobs/{job_id}/file/tc_mesh?lod=low",
            "tc_mesh_medium": f"/jobs/{job_id}/file/tc_mesh?lod=medium",
            "tc_mesh_high": f"/jobs/{job_id}/file/tc_mesh?lod=high",
            "et_mesh_low": f"/jobs/{job_id}/file/et_mesh?lod=low",
            "et_mesh_medium": f"/jobs/{job_id}/file/et_mesh?lod=medium",
            "et_mesh_high": f"/jobs/{job_id}/file/et_mesh?lod=high",
        },
        "asset_sizes": asset_sizes,
        "metadata": {
            "available_modalities": metadata.get("available_modalities", []),
            "missing_modalities": metadata.get("missing_modalities", []),
            "case_shape": preprocess.get("case_shape"),
            "uses_monai": metadata.get("uses_monai"),
            "synthesis_error": metadata.get("synthesis_error"),
        },
    }

def _run_job(
    job_id: str,
    case_id: str,
    case_dir: Path,
    enable_synthesis: bool,
    syn_steps: int,
    generate_mesh: bool,
) -> None:
    _set_job(job_id, status="running", error=None)
    seg_w = PROJECT_ROOT / "models" / "segmentation_module" / "model-weight" / "final_model_unet.pth"
    syn_w = PROJECT_ROOT / "models" / "synthesis_module" / "models"
    try:
        logger.info("[JOB %s] Starting processing for case %s", job_id, case_id)
        paths = build_modality_paths(case_id, str(case_dir))
        report = process_case(
            case_id=case_id,
            paths=paths,
            out_dir=str(OUTPUT_ROOT),
            seg_w=str(seg_w),
            syn_w=str(syn_w) if enable_synthesis else "",
            device=device,
            roi=(128, 128, 64),
            syn_steps=syn_steps,
            max_size=240,
            generate_mesh=generate_mesh,
        )
        report = to_jsonable(report)
        status = report.get("status", "completed") if isinstance(report, dict) else "completed"
        _set_job(job_id, status=status, report=report, error=None)
    except Exception as exc:
        error_msg = str(exc)
        logger.error("[JOB %s] ERROR: %s", job_id, error_msg)
        logger.error("[JOB %s] Traceback:\n%s", job_id, traceback.format_exc())
        failed_report = {"case_id": case_id, "status": "failed", "error": error_msg}
        try:
            report_path = _get_report_path(job_id)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", encoding="utf-8") as fh:
                json.dump(failed_report, fh, indent=2)
        except Exception:
            logger.error("[JOB %s] Failed writing failed report:\n%s", job_id, traceback.format_exc())
        _set_job(job_id, status="failed", error=error_msg, report=failed_report)

@app.get("/")
def root():
    return FileResponse(FRONTEND_ROOT / "index.html")

@app.get("/api/health")
def health():
    return _json_response({"status": "ok", "device": device})

@app.get("/api/debug/jobs")
def debug_jobs():
    jobs = _list_jobs_snapshot()
    return _json_response(
        {
            "total_jobs": len(jobs),
            "jobs": {
                jid: {
                    "status": job.get("status"),
                    "case_id": job.get("case_id"),
                    "has_report": isinstance(job.get("report"), dict) or _get_report_path(jid).exists(),
                    "error": job.get("error"),
                }
                for jid, job in jobs.items()
            },
        }
    )

@app.post("/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    case_id: str = Form(...),
    enable_synthesis: bool = Form(True),
    generate_mesh: bool = Form(True),
    syn_steps: int = Form(50),
    flair: UploadFile | None = File(default=None),
    t1: UploadFile | None = File(default=None),
    t1ce: UploadFile | None = File(default=None),
    t2: UploadFile | None = File(default=None),
):
    uploads = {"flair": flair, "t1": t1, "t1ce": t1ce, "t2": t2}
    if not any(v is not None for v in uploads.values()):
        raise HTTPException(status_code=400, detail="At least one modality file is required")
    job_id = str(uuid.uuid4())
    case_dir = DATA_ROOT / job_id / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    for modality, upload in uploads.items():
        if upload is None:
            continue
        suffix = ".nii.gz" if (upload.filename and upload.filename.endswith(".nii.gz")) else ".nii"
        _save_upload(upload, case_dir / f"{case_id}_{modality}{suffix}")
    _set_job(
        job_id,
        status="queued",
        case_id=case_id,
        error=None,
        options={
            "enable_synthesis": enable_synthesis,
            "generate_mesh": generate_mesh,
            "syn_steps": syn_steps,
        },
    )
    background_tasks.add_task(_run_job, job_id, case_id, case_dir, enable_synthesis, syn_steps, generate_mesh)
    return _json_response({"job_id": job_id, "status": "queued"})

@app.get("/jobs/{job_id}/status")
def get_job_status(job_id: str):
    job = _get_job_snapshot(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if "report" not in job:
        disk_report = _load_report_from_disk(job_id)
        if disk_report is not None:
            _set_job(job_id, report=disk_report)
            job = _get_job_snapshot(job_id) or job
    return _json_response(_build_status_payload(job_id, job))

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = _get_job_snapshot(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if "report" not in job:
        disk_report = _load_report_from_disk(job_id)
        if disk_report is not None:
            _set_job(job_id, report=disk_report)
            job = _get_job_snapshot(job_id) or job
    return _json_response(job)

@app.get("/jobs/{job_id}/report")
def get_report(job_id: str):
    report = _ensure_job_report_loaded(job_id)
    if not report:
        raise HTTPException(status_code=404, detail="Job not found or not finished")
    return _json_response(_summary_from_report(job_id, report))

@app.get("/jobs/{job_id}/report/full")
def get_full_report(job_id: str):
    report = _ensure_job_report_loaded(job_id)
    if not report:
        raise HTTPException(status_code=404, detail="Job not found or not finished")
    return _json_response(report)

@app.get("/jobs/{job_id}/preview/{plane}")
def get_preview(job_id: str, plane: str):
    if plane not in PREVIEW_PLANES:
        raise HTTPException(status_code=404, detail="Unknown preview plane")
    report = _ensure_job_report_loaded(job_id)
    if not report:
        raise HTTPException(status_code=404, detail="Job not found or not finished")
    preview_paths = (report.get("paths") or {}).get("preview_paths") or {}
    preview_path = _safe_path(preview_paths.get(plane))
    if preview_path is None:
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(preview_path, media_type="image/png")

@app.get("/jobs/{job_id}/file/{kind}")
def get_output_file(job_id: str, kind: str, lod: str = Query(DEFAULT_MESH_LOD)):
    report = _ensure_job_report_loaded(job_id)
    if not report:
        raise HTTPException(status_code=404, detail="Completed job not found")
    paths = report.get("paths", {})
    mesh_paths = paths.get("mesh_paths", {}) or {}
    brain_mesh_paths = paths.get("brain_mesh_paths", {}) or {}
    wt_mesh_paths = paths.get("wt_mesh_paths", {}) or {}
    tc_mesh_paths = paths.get("tc_mesh_paths", {}) or {}
    et_mesh_paths = paths.get("et_mesh_paths", {}) or {}
    mapping = {
        "pred_raw": paths.get("pred_raw_path"),
        "pred_post": paths.get("pred_post_path"),
        "report": paths.get("report_path"),
    }
    if kind == "mesh":
        selected = mesh_paths.get(lod) or mesh_paths.get(DEFAULT_MESH_LOD) or paths.get("mesh_path")
    elif kind == "brain_mesh":
        selected = brain_mesh_paths.get(lod) or brain_mesh_paths.get(DEFAULT_MESH_LOD) or paths.get("brain_mesh_path")
    elif kind == "wt_mesh":
        selected = wt_mesh_paths.get(lod) or wt_mesh_paths.get(DEFAULT_MESH_LOD) or paths.get("wt_mesh_path")
    elif kind == "tc_mesh":
        selected = tc_mesh_paths.get(lod) or tc_mesh_paths.get(DEFAULT_MESH_LOD) or paths.get("tc_mesh_path")
    elif kind == "et_mesh":
        selected = et_mesh_paths.get(lod) or et_mesh_paths.get(DEFAULT_MESH_LOD) or paths.get("et_mesh_path")
    else:
        selected = mapping.get(kind)
    output_file = _safe_path(selected)
    if output_file is None:
        raise HTTPException(status_code=404, detail="Requested file not found")
    media_type = mimetypes.guess_type(str(output_file))[0] or "application/octet-stream"
    return FileResponse(output_file, media_type=media_type, filename=output_file.name)

if __name__ == "__main__":
    import uvicorn
    print("Starting Brain Tumor Analysis API server...")
    print("Open your browser to: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
