from __future__ import annotations
import json
import logging
import mimetypes
import traceback
import uuid
import numpy as np
import nibabel as nib
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.config import (
    APP_ROOT, FRONTEND_ROOT, DATA_ROOT, PREVIEW_PLANES,
    DEFAULT_MESH_LOD, device
)
from core.utils import to_jsonable
from core.file_handler import save_upload, safe_path
from core.job_manager import job_manager
from core.report_builder import build_status_payload, summary_from_report
from core.pipeline_runner import run_job
from mesh_export import export_wt_mesh_lods, export_tc_mesh_lods, export_et_mesh_lods

logger = logging.getLogger("brain_api")
logging.basicConfig(level=logging.INFO)

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
app.mount("/js", StaticFiles(directory=str(APP_ROOT / "js")), name="js")

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

@app.get("/")
def root():
    return FileResponse(FRONTEND_ROOT / "index.html")

@app.get("/api/health")
def health():
    return _json_response({"status": "ok", "device": device})

@app.get("/api/debug/jobs")
def debug_jobs():
    jobs = job_manager.list_jobs_snapshot()
    return _json_response(
        {
            "total_jobs": len(jobs),
            "jobs": {
                jid: {
                    "status": job.get("status"),
                    "case_id": job.get("case_id"),
                    "has_report": isinstance(job.get("report"), dict) or job_manager.get_report_path(jid).exists(),
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
    syn_steps: int = Form(25),
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
        save_upload(upload, case_dir / f"{case_id}_{modality}{suffix}")
    job_manager.set_job(
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
    background_tasks.add_task(run_job, job_id, case_id, case_dir, enable_synthesis, syn_steps, generate_mesh)
    return _json_response({"job_id": job_id, "status": "queued"})

@app.get("/jobs/{job_id}/status")
def get_job_status(job_id: str):
    job = job_manager.get_job_snapshot(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if "report" not in job:
        disk_report = job_manager.load_report_from_disk(job_id)
        if disk_report is not None:
            job_manager.set_job(job_id, report=disk_report)
            job = job_manager.get_job_snapshot(job_id) or job
    return _json_response(build_status_payload(job_id, job))

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = job_manager.get_job_snapshot(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if "report" not in job:
        disk_report = job_manager.load_report_from_disk(job_id)
        if disk_report is not None:
            job_manager.set_job(job_id, report=disk_report)
            job = job_manager.get_job_snapshot(job_id) or job
    return _json_response(job)

@app.get("/jobs/{job_id}/report")
def get_report(job_id: str):
    report = job_manager.ensure_job_report_loaded(job_id)
    if not report:
        raise HTTPException(status_code=404, detail="Job not found or not finished")
    return _json_response(summary_from_report(job_id, report))

@app.get("/jobs/{job_id}/report/full")
def get_full_report(job_id: str):
    report = job_manager.ensure_job_report_loaded(job_id)
    if not report:
        raise HTTPException(status_code=404, detail="Job not found or not finished")
    return _json_response(report)

@app.get("/jobs/{job_id}/preview/{plane}")
def get_preview(job_id: str, plane: str):
    if plane not in PREVIEW_PLANES:
        raise HTTPException(status_code=404, detail="Unknown preview plane")
    report = job_manager.ensure_job_report_loaded(job_id)
    if not report:
        raise HTTPException(status_code=404, detail="Job not found or not finished")
    preview_paths = (report.get("paths") or {}).get("preview_paths") or {}
    preview_path = safe_path(preview_paths.get(plane))
    if preview_path is None:
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(preview_path, media_type="image/png")

@app.get("/jobs/{job_id}/synthesis_preview/{modality}")
def get_synthesis_preview(job_id: str, modality: str):
    report = job_manager.ensure_job_report_loaded(job_id)
    if not report:
        raise HTTPException(status_code=404, detail="Job not found or not finished")
    syn_paths = (report.get("paths") or {}).get("synthesis_preview_paths") or {}
    syn_path = safe_path(syn_paths.get(modality))
    if syn_path is None:
        raise HTTPException(status_code=404, detail=f"Synthesis preview for {modality} not found")
    return FileResponse(syn_path, media_type="image/png")

@app.get("/jobs/{job_id}/file/gt_wt_mesh")
def get_gt_wt_mesh(job_id: str, lod: str = Query(DEFAULT_MESH_LOD)):
    job = job_manager.get_job_snapshot(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    gt_wt_paths = job.get("gt_wt_paths", {})
    mesh_path = safe_path(gt_wt_paths.get(lod))
    if mesh_path is None:
        raise HTTPException(status_code=404, detail=f"GT WT mesh (lod={lod}) not found")
    return FileResponse(mesh_path, media_type="application/octet-stream", filename=mesh_path.name)

@app.get("/jobs/{job_id}/file/gt_tc_mesh")
def get_gt_tc_mesh(job_id: str, lod: str = Query(DEFAULT_MESH_LOD)):
    job = job_manager.get_job_snapshot(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    gt_tc_paths = job.get("gt_tc_paths", {})
    mesh_path = safe_path(gt_tc_paths.get(lod))
    if mesh_path is None:
        raise HTTPException(status_code=404, detail=f"GT TC mesh (lod={lod}) not found")
    return FileResponse(mesh_path, media_type="application/octet-stream", filename=mesh_path.name)

@app.get("/jobs/{job_id}/file/gt_et_mesh")
def get_gt_et_mesh(job_id: str, lod: str = Query(DEFAULT_MESH_LOD)):
    job = job_manager.get_job_snapshot(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    gt_et_paths = job.get("gt_et_paths", {})
    mesh_path = safe_path(gt_et_paths.get(lod))
    if mesh_path is None:
        raise HTTPException(status_code=404, detail=f"GT ET mesh (lod={lod}) not found")
    return FileResponse(mesh_path, media_type="application/octet-stream", filename=mesh_path.name)

@app.get("/jobs/{job_id}/file/{kind}")
def get_output_file(job_id: str, kind: str, lod: str = Query(DEFAULT_MESH_LOD)):
    report = job_manager.ensure_job_report_loaded(job_id)
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
    output_file = safe_path(selected)
    if output_file is None:
        raise HTTPException(status_code=404, detail="Requested file not found")
    media_type = mimetypes.guess_type(str(output_file))[0] or "application/octet-stream"
    return FileResponse(output_file, media_type=media_type, filename=output_file.name)

@app.post("/jobs/{job_id}/ground_truth")
async def upload_ground_truth(
    job_id: str,
    seg_file: UploadFile = File(...),
):
    job = job_manager.get_job_snapshot(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    case_out = job_manager.get_case_dir(job_id)
    case_out.mkdir(parents=True, exist_ok=True)
    gt_dir = case_out / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    gt_nifti_path = gt_dir / "gt_seg.nii.gz"
    save_upload(seg_file, gt_nifti_path)

    try:
        gt_img = nib.load(str(gt_nifti_path))
        gt_data = gt_img.get_fdata().astype(np.uint8)
        gt_data[gt_data == 4] = 3

        logger.info("[GT] Loaded ground truth mask shape=%s, unique=%s", gt_data.shape, np.unique(gt_data).tolist())

        gt_prefix = str(gt_dir / f"{job_id}_gt")
        gt_wt_paths = export_wt_mesh_lods(gt_data, f"{gt_prefix}_wt")
        gt_tc_paths = export_tc_mesh_lods(gt_data, f"{gt_prefix}_tc")
        gt_et_paths = export_et_mesh_lods(gt_data, f"{gt_prefix}_et")

        gt_info = {
            "status": "ok",
            "shape": list(gt_data.shape),
            "unique_labels": np.unique(gt_data).tolist(),
            "voxel_counts": {
                "WT": int((gt_data > 0).sum()),
                "TC": int(np.isin(gt_data, [1, 3]).sum()),
                "ET": int((gt_data == 3).sum()),
            },
            "meshes": {
                "wt": {
                    lod: f"/jobs/{job_id}/file/gt_wt_mesh?lod={lod}"
                    for lod in ("low", "medium", "high")
                    if gt_wt_paths.get(lod)
                },
                "tc": {
                    lod: f"/jobs/{job_id}/file/gt_tc_mesh?lod={lod}"
                    for lod in ("low", "medium", "high")
                    if gt_tc_paths.get(lod)
                },
                "et": {
                    lod: f"/jobs/{job_id}/file/gt_et_mesh?lod={lod}"
                    for lod in ("low", "medium", "high")
                    if gt_et_paths.get(lod)
                },
            },
        }

        job_manager.set_job(job_id, gt_wt_paths=gt_wt_paths, gt_tc_paths=gt_tc_paths, gt_et_paths=gt_et_paths)

        return _json_response(gt_info)

    except Exception as exc:
        logger.error("[GT] Failed processing ground truth: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to process ground truth mask: {exc}")

if __name__ == "__main__":
    from typing import Any
    import uvicorn
    print("Starting Brain Tumor Analysis API server...")
    print("Open your browser to: http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
