import json
import logging
import traceback
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from core.config import OUTPUT_ROOT
from core.utils import to_jsonable

logger = logging.getLogger("job_manager")

class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def set_job(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            current = self._jobs.get(job_id, {}).copy()
            current.update(updates)
            self._jobs[job_id] = current

    def get_job_snapshot(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def list_jobs_snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {jid: dict(job) for jid, job in self._jobs.items()}

    def get_case_dir(self, job_id: str) -> Path:
        return OUTPUT_ROOT / job_id

    def get_report_path(self, job_id: str) -> Path:
        return self.get_case_dir(job_id) / f"{job_id}_report.json"

    def load_report_from_disk(self, job_id: str) -> Optional[Dict[str, Any]]:
        report_path = self.get_report_path(job_id)
        if not report_path.exists():
            return None
        try:
            with report_path.open("r", encoding="utf-8") as fh:
                return to_jsonable(json.load(fh))
        except Exception:
            logger.error("[JOB %s] Failed reading report from disk:\n%s", job_id, traceback.format_exc())
            return None

    def ensure_job_report_loaded(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.get_job_snapshot(job_id)
        if not job:
            return None
        if isinstance(job.get("report"), dict):
            return job["report"]
        report = self.load_report_from_disk(job_id)
        if report is not None:
            self.set_job(job_id, report=report)
        return report

job_manager = JobManager()
