from typing import Any, Dict
from core.config import PREVIEW_PLANES, DEFAULT_MESH_LOD
from core.file_handler import file_size
from core.job_manager import job_manager

def build_status_payload(job_id: str, job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "case_id": job.get("case_id"),
        "status": job.get("status", "queued"),
        "error": job.get("error"),
        "has_report": isinstance(job.get("report"), dict) or job_manager.get_report_path(job_id).exists(),
    }

def summary_from_report(job_id: str, report: Dict[str, Any]) -> Dict[str, Any]:
    paths = report.get("paths", {}) if isinstance(report, dict) else {}
    preview_paths = paths.get("preview_paths", {}) or {}
    synthesis_preview_paths = paths.get("synthesis_preview_paths", {}) or {}
    mesh_paths = paths.get("mesh_paths", {}) or {}
    brain_mesh_paths = paths.get("brain_mesh_paths", {}) or {}
    wt_mesh_paths = paths.get("wt_mesh_paths", {}) or {}
    tc_mesh_paths = paths.get("tc_mesh_paths", {}) or {}
    et_mesh_paths = paths.get("et_mesh_paths", {}) or {}
    metadata = report.get("metadata", {}) or {}
    preprocess = metadata.get("preprocess", {}) or {}
    
    asset_sizes = {
        "prediction_bytes": file_size(paths.get("pred_post_path")),
        "report_bytes": file_size(paths.get("report_path")),
        "mesh_low_bytes": file_size(mesh_paths.get("low") or paths.get("mesh_path")),
        "mesh_medium_bytes": file_size(mesh_paths.get("medium")),
        "mesh_high_bytes": file_size(mesh_paths.get("high")),
        "brain_mesh_low_bytes": file_size(brain_mesh_paths.get("low") or paths.get("brain_mesh_path")),
        "brain_mesh_medium_bytes": file_size(brain_mesh_paths.get("medium")),
        "brain_mesh_high_bytes": file_size(brain_mesh_paths.get("high")),
        "wt_mesh_low_bytes": file_size(wt_mesh_paths.get("low") or paths.get("wt_mesh_path")),
        "wt_mesh_medium_bytes": file_size(wt_mesh_paths.get("medium")),
        "wt_mesh_high_bytes": file_size(wt_mesh_paths.get("high")),
        "tc_mesh_low_bytes": file_size(tc_mesh_paths.get("low") or paths.get("tc_mesh_path")),
        "tc_mesh_medium_bytes": file_size(tc_mesh_paths.get("medium")),
        "tc_mesh_high_bytes": file_size(tc_mesh_paths.get("high")),
        "et_mesh_low_bytes": file_size(et_mesh_paths.get("low") or paths.get("et_mesh_path")),
        "et_mesh_medium_bytes": file_size(et_mesh_paths.get("medium")),
        "et_mesh_high_bytes": file_size(et_mesh_paths.get("high")),
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
        "synthesis_preview": {
            mod: f"/jobs/{job_id}/synthesis_preview/{mod}"
            for mod, p in synthesis_preview_paths.items()
            if p
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
