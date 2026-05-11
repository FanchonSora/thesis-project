import shutil
from pathlib import Path
from typing import Optional
from fastapi import UploadFile

def safe_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None

def file_size(path_str: Optional[str]) -> Optional[int]:
    path = safe_path(path_str)
    return path.stat().st_size if path else None

def save_upload(upload: UploadFile, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh)
