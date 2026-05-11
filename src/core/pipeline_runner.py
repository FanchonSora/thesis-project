import json
import logging
import traceback
from pathlib import Path

from core.config import PROJECT_ROOT, OUTPUT_ROOT, device
from core.job_manager import job_manager
from core.utils import to_jsonable
from preprocessing import build_modality_paths
from run_pipeline import process_case

logger = logging.getLogger("pipeline_runner")

def run_job(
    job_id: str,
    case_id: str,
    case_dir: Path,
    enable_synthesis: bool,
    syn_steps: int,
    generate_mesh: bool,
) -> None:
    job_manager.set_job(job_id, status="running", error=None)
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
        job_manager.set_job(job_id, status=status, report=report, error=None)
        
    except Exception as exc:
        error_msg = str(exc)
        logger.error("[JOB %s] ERROR: %s", job_id, error_msg)
        logger.error("[JOB %s] Traceback:\n%s", job_id, traceback.format_exc())
        failed_report = {"case_id": case_id, "status": "failed", "error": error_msg}
        
        try:
            report_path = job_manager.get_report_path(job_id)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", encoding="utf-8") as fh:
                json.dump(failed_report, fh, indent=2)
        except Exception:
            logger.error("[JOB %s] Failed writing failed report:\n%s", job_id, traceback.format_exc())
            
        job_manager.set_job(job_id, status="failed", error=error_msg, report=failed_report)
