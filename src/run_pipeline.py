from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch not found.") from exc
# Path setup
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CURRENT_DIR))
# Module imports
from pipeline_types import PipelineConfig, PipelinePaths, PipelineResult, to_jsonable
from preprocessing import (
    MODALITY_ORDER,
    build_modality_paths,
    load_raw_volumes,
    apply_segmentation_preprocessing,
    preprocess_case,
    save_nifti,
)
from segmentation import HAS_MONAI, load_trained_model, post_process, run_segmentation
from synthesis import SynthesisWrapper
from mesh_export import (
    export_brain_mesh_lods,
    export_et_mesh_lods,
    export_tc_mesh_lods,
    export_wt_mesh_lods,
)
from visualization import (
    compute_region_volumes,
    save_preview_images,
    save_synthesis_previews,
)
LOGGER = logging.getLogger("brain_tumor_pipeline")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def process_case(
    case_id: str,
    paths: Dict[str, str],
    out_dir: str,
    seg_w: str,
    syn_w: str = "",
    device: str = "cpu",
    roi: Optional[Tuple[int, int, int]] = None,
    syn_steps: int = 25,
    max_size: int = 240,
    post_min_size: int = 100,
    generate_mesh: bool = True,
) -> Dict[str, Any]:
    case_out = Path(out_dir) / case_id
    case_out.mkdir(parents=True, exist_ok=True)
    errors = []
    config = PipelineConfig(
        seg_w=seg_w,
        syn_w=syn_w,
        device=device,
        roi=roi,
        syn_steps=syn_steps,
        max_size=max_size,
        post_min_size=post_min_size,
        generate_mesh=generate_mesh,
    )
    try:
        # ================================================================
        # Determine if synthesis is needed
        # ================================================================
        missing_flags = {m: 0 if (paths.get(m) and os.path.exists(paths[m])) else 1 for m in MODALITY_ORDER}
        has_missing = any(missing_flags.values())
        needs_synthesis = has_missing and syn_w

        if needs_synthesis:
            # ==============================================================
            # PATH A: Missing modalities → Synthesis then Segmentation
            #
            # Flow:
            #   1. Load raw volumes (NO normalization)
            #   2. Create copies of 3 available raw modalities
            #   3. Apply synthesis preprocessing (percentile → [-1,1])
            #      and run synthesis model → get missing modality in [0,1]
            #   4. Combine 3 original raw modalities + synthesized output
            #   5. Apply segmentation preprocessing (z-score) to all 4
            #   6. Feed into segmentation model
            # ==============================================================
            LOGGER.info("[1/4] Loading raw volumes (no normalization) for case=%s", case_id)
            raw_result = load_raw_volumes(paths=paths, max_size=max_size)

            # Stage 2: Synthesis from raw volumes
            LOGGER.info("[2/4] Synthesis from raw volumes")
            synthesis = SynthesisWrapper(syn_w, device=device)
            synthesized_outputs = synthesis.synthesize_from_raw(
                raw_vols=raw_result.raw_vols,
                missing_modalities=raw_result.missing_modalities,
                num_steps=syn_steps,
            )
            LOGGER.info("[syn] synthesis_status=%s", synthesis.status)

            # Merge: 3 original raw + synthesized outputs into a complete raw_vols dict
            complete_raw_vols = {}
            for mod in MODALITY_ORDER:
                if raw_result.raw_vols.get(mod) is not None:
                    # Use original raw volume (unprocessed)
                    complete_raw_vols[mod] = raw_result.raw_vols[mod]
                elif mod in synthesized_outputs:
                    # Use synthesized output (already in [0,1] raw-like scale)
                    complete_raw_vols[mod] = synthesized_outputs[mod]
                else:
                    # Should not happen, but fill with zeros as safety
                    complete_raw_vols[mod] = np.zeros(raw_result.case_shape, dtype=np.float32)

            # Stage 3: Apply segmentation preprocessing to all 4 modalities
            LOGGER.info("[3/4] Applying segmentation preprocessing (z-score) to all 4 modalities")
            stacked = apply_segmentation_preprocessing(
                raw_vols=complete_raw_vols,
                case_shape=raw_result.case_shape,
                lower=0.5,
                upper=99.9,
            )

            # Build PreprocessResult-like info for downstream use
            brain_mask = raw_result.brain_mask
            affine = raw_result.affine
            ds_factor = raw_result.ds_factor
            case_shape = raw_result.case_shape
            available_modalities = raw_result.available_modalities
            missing_modalities = raw_result.missing_modalities
            per_modality_info = raw_result.per_modality_info

        else:
            # ==============================================================
            # PATH B: No missing modalities OR no synthesis weights
            #         → Direct segmentation preprocessing (legacy path)
            # ==============================================================
            LOGGER.info("[1/4] Preprocessing case=%s (all modalities available)", case_id)
            prep = preprocess_case(paths=paths, lower=0.5, upper=99.9, max_size=max_size)
            stacked = prep.stacked
            brain_mask = prep.brain_mask
            affine = prep.affine
            ds_factor = prep.ds_factor
            case_shape = prep.case_shape
            available_modalities = prep.available_modalities
            missing_modalities = prep.missing_modalities
            per_modality_info = prep.per_modality_info

            # Synthesis stage
            LOGGER.info("[2/4] Synthesis")
            synthesis = SynthesisWrapper(syn_w, device=device)
            if syn_w and has_missing:
                # This case means has_missing but no syn_w was handled above,
                # so here means: has_missing=False or no syn_w
                stacked = synthesis.synthesize(
                    stacked,
                    [missing_flags[m] for m in MODALITY_ORDER],
                    num_steps=syn_steps,
                )
                LOGGER.info("[syn] Final synthesis_status=%s", synthesis.status)
            else:
                synthesis.status = "skipped"
                if not syn_w:
                    synthesis.error = "Skipped because syn_w is empty"
                elif not any(missing_flags.values()):
                    synthesis.error = "Skipped because no modality is missing"
                LOGGER.info("[syn] Skipped. Reason: %s", synthesis.error)

        # Stage 3 (continued): Segmentation
        LOGGER.info("[3/4] Segmentation")
        model, _ = load_trained_model(seg_w)
        pred_raw = run_segmentation(model, stacked, device=device, roi=roi)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pred_post = post_process(pred_raw, brain_mask=brain_mask, min_size=post_min_size)
        # Stage 4: Save outputs
        LOGGER.info("[4/4] Saving outputs")
        # NIfTI predictions
        pred_raw_path = str(case_out / f"{case_id}_pred_raw.nii.gz")
        pred_post_path = str(case_out / f"{case_id}_pred_post.nii.gz")
        pred_compat_path = str(case_out / f"{case_id}_pred.nii.gz")
        if config.save_raw_prediction:
            save_nifti(pred_raw_path, pred_raw, affine)
        if config.save_post_prediction:
            save_nifti(pred_post_path, pred_post, affine)
            save_nifti(pred_compat_path, pred_post, affine)
        # Preview images
        base_volume = stacked[0] if stacked.ndim == 4 else np.asarray(stacked)
        preview_paths = save_preview_images(case_id, case_out, pred_post, base_volume)
        synthesis_preview_paths = save_synthesis_previews(
            case_id, case_out, stacked, missing_flags, synthesis.status,
        )
        # Meshes
        empty_lods = {"low": None, "medium": None, "high": None}
        mesh_paths = export_wt_mesh_lods(pred_post, str(case_out / f"{case_id}_wt")) if generate_mesh else dict(empty_lods)
        brain_mesh_paths = export_brain_mesh_lods(brain_mask, str(case_out / f"{case_id}_brain")) if generate_mesh else dict(empty_lods)
        wt_mesh_paths = export_wt_mesh_lods(pred_post, str(case_out / f"{case_id}_wt_region")) if generate_mesh else dict(empty_lods)
        tc_mesh_paths = export_tc_mesh_lods(pred_post, str(case_out / f"{case_id}_tc_region")) if generate_mesh else dict(empty_lods)
        et_mesh_paths = export_et_mesh_lods(pred_post, str(case_out / f"{case_id}_et_region")) if generate_mesh else dict(empty_lods)
        # Volume metrics
        voxel_counts, mm3 = compute_region_volumes(pred_post, affine)
        # Build result
        result = PipelineResult(
            case_id=case_id,
            status="completed",
            paths=PipelinePaths(
                pred_raw_path=pred_raw_path,
                pred_post_path=pred_post_path,
                mesh_path=mesh_paths.get("low") or mesh_paths.get("high"),
                mesh_paths=mesh_paths,
                brain_mesh_path=brain_mesh_paths.get("low") or brain_mesh_paths.get("high"),
                brain_mesh_paths=brain_mesh_paths,
                wt_mesh_path=wt_mesh_paths.get("low") or wt_mesh_paths.get("high"),
                wt_mesh_paths=wt_mesh_paths,
                tc_mesh_path=tc_mesh_paths.get("low") or tc_mesh_paths.get("high"),
                tc_mesh_paths=tc_mesh_paths,
                et_mesh_path=et_mesh_paths.get("low") or et_mesh_paths.get("high"),
                et_mesh_paths=et_mesh_paths,
                preview_paths=preview_paths,
                synthesis_preview_paths=synthesis_preview_paths,
                report_path=str(case_out / f"{case_id}_report.json"),
            ),
            missing_flags=missing_flags,
            synthesis_status=synthesis.status,
            downsample_factor=float(ds_factor),
            region_volumes_voxels=voxel_counts,
            region_volumes_mm3=mm3,
            affine=affine.tolist() if affine is not None else None,
            errors=errors,
            metadata={
                "available_modalities": available_modalities,
                "missing_modalities": missing_modalities,
                "preprocess": {
                    "ds_factor": float(ds_factor),
                    "case_shape": list(case_shape),
                    "affine": affine.tolist() if affine is not None else None,
                    "per_modality_info": to_jsonable(
                        {
                            k: {
                                "path": v.path,
                                "original_shape": list(v.original_shape),
                                "processed_shape": list(v.processed_shape),
                                "voxel_spacing_mm": list(v.voxel_spacing_mm),
                                "affine": v.affine.tolist() if v.affine is not None else None,
                            }
                            for k, v in per_modality_info.items()
                        }
                    ),
                },
                "config": to_jsonable(asdict(config)),
                "synthesis_error": synthesis.error,
                "synthesis_loaded_targets": synthesis.loaded_targets,
                "uses_monai": HAS_MONAI,
            },
        )
        result_dict = to_jsonable(result.to_dict())
        with open(result.paths.report_path, "w", encoding="utf-8") as fh:
            json.dump(result_dict, fh, indent=2)
        return result_dict
    except Exception as exc:
        errors.append(traceback.format_exc())
        failed = to_jsonable({"case_id": case_id, "status": "failed", "error": str(exc), "errors": errors})
        with open(case_out / f"{case_id}_report.json", "w", encoding="utf-8") as fh:
            json.dump(failed, fh, indent=2)
        return failed


# CLI entry point
def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name == "src" else script_dir
    parser = argparse.ArgumentParser(description="Brain tumor inference pipeline")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", default=str(project_root / "output"))
    parser.add_argument(
        "--seg-w",
        default=str(project_root / "models" / "segmentation_module" / "model-weight" / "final_model_unet.pth"),
    )
    parser.add_argument(
        "--syn-w",
        default=str(project_root / "models" / "synthesis_module" / "models"),
        help="Path to synthesis models directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu", "mps"],
    )
    parser.add_argument("--roi", nargs=3, type=int, metavar=("X", "Y", "Z"))
    parser.add_argument("--syn-steps", type=int, default=25)
    parser.add_argument("--max-size", type=int, default=240)
    parser.add_argument("--post-min-size", type=int, default=100)
    parser.add_argument("--no-mesh", action="store_true")
    args = parser.parse_args()
    paths = build_modality_paths(args.case_id, args.input_dir)
    report = process_case(
        case_id=args.case_id,
        paths=paths,
        out_dir=args.out_dir,
        seg_w=args.seg_w,
        syn_w=args.syn_w,
        device=args.device,
        roi=tuple(args.roi) if args.roi else None,
        syn_steps=args.syn_steps,
        max_size=args.max_size,
        post_min_size=args.post_min_size,
        generate_mesh=not args.no_mesh,
    )
    raise SystemExit(0 if report.get("status") == "completed" else 1)

if __name__ == "__main__":
    main()
