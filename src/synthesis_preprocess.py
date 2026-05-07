from __future__ import annotations
import numpy as np

def normalize_for_synthesis(
    volumes: np.ndarray,
    percentile_low: float = 0.5,
    percentile_high: float = 99.5,
) -> np.ndarray:
    out = volumes.copy().astype(np.float32)
    for ch in range(out.shape[0]):
        vol = out[ch]
        mask = vol > 0
        if np.any(mask):
            lo = float(np.percentile(vol[mask], percentile_low))
            hi = float(np.percentile(vol[mask], percentile_high))
            vol = np.clip(vol, lo, hi)
            if hi > lo:
                vol = (vol - lo) / (hi - lo)
            else:
                vol = np.zeros_like(vol)
        else:
            vol = np.zeros_like(vol)
        out[ch] = vol * 2.0 - 1.0
    return out

def denormalize_from_synthesis(vol: np.ndarray) -> np.ndarray:
    out = (vol + 1.0) / 2.0
    return np.clip(out, 0.0, 1.0)

def postprocess_synthesized_to_zscore(vol: np.ndarray) -> np.ndarray:
    out = vol.copy().astype(np.float32)
    mask = out > 0.01
    if np.any(mask):
        mean = float(out[mask].mean())
        std = float(out[mask].std())
        if std > 1e-7:
            out[mask] = (out[mask] - mean) / std
        else:
            out[mask] = out[mask] - mean
    return out
