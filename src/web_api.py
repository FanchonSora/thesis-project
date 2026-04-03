from __future__ import annotations

import json
import logging
import shutil
import traceback
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .preprocessing import build_modality_paths
from .run_pipeline import process_case


logger = logging.getLogger("brain_api")
logging.basicConfig(level=logging.INFO)

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
DATA_ROOT = APP_ROOT / "web_data" / "uploads"
OUTPUT_ROOT = APP_ROOT / "web_output"
STATIC_ROOT = APP_ROOT / "web_data"

DATA_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

app = FastAPI(
    title="Brain Tumor Analysis API",
    description="Advanced MRI brain tumor segmentation platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = Lock()


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


def _get_case_dir(job_id: str) -> Path:
    return OUTPUT_ROOT / job_id


def _get_report_path(job_id: str) -> Path:
    return _get_case_dir(job_id) / f"{job_id}_report.json"


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


def _load_report_from_disk(job_id: str) -> Optional[Dict[str, Any]]:
    report_path = _get_report_path(job_id)
    if not report_path.exists():
        return None

    try:
        with report_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return to_jsonable(data)
    except Exception:
        logger.error("[JOB %s] Failed reading report from disk:\n%s", job_id, traceback.format_exc())
        return None


def _ensure_job_report_loaded(job_id: str) -> Optional[Dict[str, Any]]:
    job = _get_job_snapshot(job_id)
    if not job:
        return None

    report = job.get("report")
    if isinstance(report, dict):
        return report

    disk_report = _load_report_from_disk(job_id)
    if disk_report is not None:
        _set_job(job_id, report=disk_report)
        return disk_report

    return None


def _build_status_payload(job_id: str, job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "case_id": job.get("case_id"),
        "status": job.get("status", "queued"),
        "error": job.get("error"),
        "has_report": isinstance(job.get("report"), dict) or _get_report_path(job_id).exists(),
    }


def _run_job(job_id: str, case_id: str, case_dir: Path) -> None:
    _set_job(job_id, status="running", error=None)

    seg_w = PROJECT_ROOT / "models" / "segmentation-module" / "model-weight" / "final_model_unet.pth"
    syn_w = PROJECT_ROOT / "models" / "synthesis-module" / "model-weight" / "epoch_118.pth"

    try:
        logger.info("[JOB %s] Starting processing for case %s", job_id, case_id)
        logger.info("[JOB %s] Building paths...", job_id)

        paths = build_modality_paths(case_id, str(case_dir))

        logger.info("[JOB %s] Calling process_case...", job_id)
        report = process_case(
            case_id=case_id,
            paths=paths,
            out_dir=str(OUTPUT_ROOT),
            seg_w=str(seg_w),
            syn_w=str(syn_w),
            device=device,
            roi=(128, 128, 64),
            syn_steps=50,
            max_size=240,
        )

        logger.info(
            "[JOB %s] process_case returned. type=%s keys=%s",
            job_id,
            type(report),
            list(report.keys()) if isinstance(report, dict) else "N/A",
        )

        report = to_jsonable(report)
        status = report.get("status", "completed") if isinstance(report, dict) else "completed"

        _set_job(
            job_id,
            status=status,
            report=report,
            error=None,
        )

        logger.info("[JOB %s] COMPLETED with status=%s", job_id, status)

    except Exception as exc:
        error_msg = str(exc)
        logger.error("[JOB %s] ERROR: %s", job_id, error_msg)
        logger.error("[JOB %s] Traceback:\n%s", job_id, traceback.format_exc())

        failed_report = {
            "case_id": case_id,
            "status": "failed",
            "error": error_msg,
        }

        try:
            report_path = _get_report_path(job_id)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", encoding="utf-8") as fh:
                json.dump(failed_report, fh, indent=2)
        except Exception:
            logger.error("[JOB %s] Failed writing failed report:\n%s", job_id, traceback.format_exc())

        _set_job(
            job_id,
            status="failed",
            error=error_msg,
            report=failed_report,
        )


@app.get("/")
def root():
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/api/health")
def health():
    return _json_response(
        {
            "status": "ok",
            "timestamp": str(Path(__file__).stat().st_mtime),
            "device": device,
        }
    )


@app.get("/api/debug/jobs")
def debug_jobs():
    jobs = _list_jobs_snapshot()
    payload = {
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
    return _json_response(payload)


@app.post("/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    case_id: str = Form(...),
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
    )

    background_tasks.add_task(_run_job, job_id, case_id, case_dir)

    return _json_response(
        {
            "job_id": job_id,
            "status": "queued",
        }
    )


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
    job = _get_job_snapshot(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    report = _ensure_job_report_loaded(job_id)
    if not report:
        raise HTTPException(status_code=409, detail="Job not finished")

    return _json_response(report)


@app.get("/jobs/{job_id}/file/{kind}")
def get_output_file(job_id: str, kind: str):
    report = _ensure_job_report_loaded(job_id)
    if not report:
        raise HTTPException(status_code=404, detail="Completed job not found")

    paths = report.get("paths", {})
    mapping = {
        "pred_raw": paths.get("pred_raw_path"),
        "pred_post": paths.get("pred_post_path"),
        "mesh": paths.get("mesh_path"),
        "report": paths.get("report_path"),
    }

    output_path = mapping.get(kind)
    if not output_path:
        raise HTTPException(status_code=404, detail="Requested file not found")

    output_file = Path(output_path)
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Requested file not found")

    return FileResponse(output_file)