from __future__ import annotations
import logging
import os
from typing import Dict, Optional, Sequence
import numpy as np
import scipy.ndimage
import torch
from preprocessing import MODALITY_ORDER
from synthesis_preprocess import (
    denormalize_from_synthesis,
    normalize_for_synthesis,
)
try:
    from models.synthesis_module.diffusion_model.trainer_brats import GaussianDiffusion
    from models.synthesis_module.diffusion_model.unet_brats import create_model
except ImportError:
    GaussianDiffusion = None
    create_model = None

LOGGER = logging.getLogger("brain_tumor_pipeline")
SYNTHESIS_IMAGE_SIZE = 128
SYNTHESIS_DEPTH_SIZE = 144

class SynthesisWrapper:
    # Mapping: target_modality → (checkpoint_filename, condition_modalities)
    MODEL_CONFIG = {
        "flair": ("model_flair_from_t1_t1ce_t2.pt", ["t1", "t1ce", "t2"]),
        "t1": ("model_t1_from_t1ce_t2_flair.pt", ["t1ce", "t2", "flair"]),
        "t1ce": ("model_t1ce_from_t1_t2_flair.pt", ["t1", "t2", "flair"]),
        "t2": ("model_t2_from_t1_t1ce_flair.pt", ["t1", "t1ce", "flair"]),
    }

    def __init__(self, models_dir: Optional[str], device: str = "cpu"):
        self.device = device
        self.models: Dict[str, torch.nn.Module] = {}
        self.diffusions: Dict[str, torch.nn.Module] = {}
        self.status = "skipped"
        self.error: Optional[str] = None
        self.loaded_targets: list[str] = []
        if not models_dir:
            self.status = "skipped"
            self.error = "Synthesis disabled because syn_w is empty"
            LOGGER.info("[syn] Disabled: syn_w is empty")
            return
        if not os.path.exists(models_dir):
            self.status = "fallback_mean"
            self.error = f"Synthesis models directory not found: {models_dir}"
            LOGGER.warning("[syn] %s", self.error)
            return
        if create_model is None or GaussianDiffusion is None:
            self.status = "fallback_mean"
            self.error = "Could not import synthesis modules"
            LOGGER.exception("[syn] Failed importing synthesis modules")
            return
        LOGGER.info("[syn] Looking for synthesis weights in %s", models_dir)
        for target, (model_file, _) in self.MODEL_CONFIG.items():
            self._try_load_model(models_dir, target, model_file)
        if self.models:
            self.status = "ready"
            LOGGER.info("[syn] Ready. Loaded targets: %s", self.loaded_targets)
        else:
            self.status = "fallback_mean"
            if self.error is None:
                self.error = "No synthesis models loaded successfully"
            LOGGER.warning("[syn] %s", self.error)

    def _try_load_model(self, models_dir: str, target: str, model_file: str) -> None:
        model_path = os.path.join(models_dir, model_file)
        if not os.path.exists(model_path):
            LOGGER.warning("[syn] Missing weight for %s: %s", target, model_path)
            return
        try:
            # Load checkpoint to CPU first to avoid CUDA OOM
            ckpt = torch.load(model_path, map_location="cpu")
            if isinstance(ckpt, dict) and "ema" in ckpt:
                state_dict = ckpt["ema"]
            elif isinstance(ckpt, dict) and "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
            ckpt_timesteps = 1000  # default fallback
            if isinstance(state_dict, dict) and "betas" in state_dict:
                ckpt_timesteps = state_dict["betas"].shape[0]
            LOGGER.info("[syn] Detected timesteps=%d from checkpoint for target=%s", ckpt_timesteps, target)
            model = create_model(
                image_size=SYNTHESIS_IMAGE_SIZE,
                num_channels=64,
                num_res_blocks=2,
                in_channels=4,   # 3 condition channels + noisy target channel
                out_channels=1,
            )
            diffusion = GaussianDiffusion(
                model,
                image_size=SYNTHESIS_IMAGE_SIZE,
                depth_size=SYNTHESIS_DEPTH_SIZE,
                timesteps=ckpt_timesteps,
                loss_type="l2",
                with_condition=True,
                channels=1,
            )  
            diffusion.load_state_dict(state_dict, strict=False)
            diffusion.eval()
            self.models[target] = model
            self.diffusions[target] = diffusion
            self.loaded_targets.append(target)
            LOGGER.info("[syn] Loaded model for target=%s on CPU (timesteps=%d)", target, ckpt_timesteps)
            del ckpt, state_dict
        except RuntimeError as exc:
            if "PytorchStreamReader" in str(exc) or "central directory" in str(exc):
                LOGGER.error("[syn] Model file for %s is CORRUPT: %s. Please re-download or re-train.", target, model_path)
                self.error = f"Corrupt model file for {target}: {model_path}"
            else:
                LOGGER.exception("[syn] Failed to load model for %s", target)
                self.error = f"Failed to load model for {target}: {exc}"
        except Exception as exc:
            LOGGER.exception("[syn] Failed to load model for %s", target)
            self.error = f"Failed to load model for {target}: {exc}"

    # ------------------------------------------------------------------
    # Synthesis from raw volumes
    # ------------------------------------------------------------------

    def synthesize_from_raw(
        self,
        raw_vols: Dict[str, "np.ndarray | None"],
        missing_modalities: list[str],
        num_steps: int = 50,
    ) -> Dict[str, np.ndarray]:
        """Synthesize missing modalities from RAW (un-normalized) available volumes.

        This method:
        1. Takes 3 raw available modality volumes (no preprocessing applied)
        2. Creates copies of these raw volumes
        3. Applies synthesis-specific preprocessing (percentile → [-1,1])
        4. Runs the diffusion model to generate the missing modality
        5. Returns the synthesized output in raw intensity scale (denormalized [0,1])

        Args:
            raw_vols: Dict mapping modality name to raw numpy array (H, W, D), or None if missing.
            missing_modalities: List of modality names that need synthesis.
            num_steps: Number of diffusion sampling steps.

        Returns:
            Dict mapping each synthesized modality name to its raw-scale output (H, W, D).
        """
        synthesized_outputs: Dict[str, np.ndarray] = {}

        if not missing_modalities:
            self.status = "skipped"
            LOGGER.info("[syn] No missing modalities. Nothing to synthesize.")
            return synthesized_outputs

        if not self.models:
            LOGGER.warning("[syn] No diffusion models available. Falling back to mean fill.")
            for target_mod in missing_modalities:
                synthesized_outputs[target_mod] = self._fallback_fill_raw(raw_vols, target_mod)
            self.status = "fallback_mean"
            return synthesized_outputs

        any_success = False
        for target_mod in missing_modalities:
            LOGGER.info("[syn] Synthesizing target=%s from raw volumes", target_mod)

            if target_mod not in self.diffusions:
                LOGGER.warning("[syn] No model for target=%s. Using fallback mean.", target_mod)
                synthesized_outputs[target_mod] = self._fallback_fill_raw(raw_vols, target_mod)
                continue

            try:
                synthesized = self._synthesize_one_from_raw(raw_vols, target_mod)
                synthesized_outputs[target_mod] = synthesized
                any_success = True
                LOGGER.info("[syn] Completed target=%s", target_mod)
            except Exception as exc:
                LOGGER.exception("[syn] Failed generating target=%s. Using fallback mean.", target_mod)
                self.error = f"Synthesis failed for {target_mod}: {exc}"
                synthesized_outputs[target_mod] = self._fallback_fill_raw(raw_vols, target_mod)

        self.status = "success" if any_success else "fallback_mean"
        return synthesized_outputs

    def _synthesize_one_from_raw(
        self,
        raw_vols: Dict[str, "np.ndarray | None"],
        target_mod: str,
    ) -> np.ndarray:
        """Synthesize one missing modality from raw input volumes.

        Steps:
        1. Copy 3 raw condition modality volumes
        2. Resize to synthesis model input size (128x128x144)
        3. Apply synthesis preprocessing (percentile normalization → [-1, 1])
        4. Run diffusion model
        5. Denormalize output to [0, 1] scale
        6. Resize back to original shape
        """
        cond_mods = self.MODEL_CONFIG[target_mod][1]

        # Step 1: Get copies of raw condition volumes
        cond_raw = []
        for m in cond_mods:
            vol = raw_vols.get(m)
            if vol is None:
                raise ValueError(f"Condition modality '{m}' is missing but needed for synthesizing '{target_mod}'")
            cond_raw.append(vol.copy())

        # Remember original shape for resizing back later
        original_shape = cond_raw[0].shape  # (H, W, D)

        # Step 2: Resize each raw condition volume to model input size
        target_shape = (SYNTHESIS_IMAGE_SIZE, SYNTHESIS_IMAGE_SIZE, SYNTHESIS_DEPTH_SIZE)
        cond_resized = np.stack(
            [
                scipy.ndimage.zoom(
                    v,
                    (
                        target_shape[0] / v.shape[0],
                        target_shape[1] / v.shape[1],
                        target_shape[2] / v.shape[2],
                    ),
                    order=1,
                )
                for v in cond_raw
            ],
            axis=0,
        )  # (3, H', W', D')

        # Step 3: Apply synthesis-specific preprocessing (percentile → [-1, 1])
        # This matches the training pipeline in dataset_brats.py NiftiPairImageGenerator3to1
        cond_resized = normalize_for_synthesis(cond_resized)

        # Step 4: Convert to tensor and run diffusion
        # (3, H, W, D) → (1, 3, D, H, W)
        cond_tensor = (
            torch.from_numpy(cond_resized)
            .float()
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
        )
        gen = self._run_diffusion_sampling(target_mod, cond_tensor)

        # Step 5: Post-process: [-1,1] → [0,1]
        synthesized = gen[0, 0].detach().cpu().numpy()  # (D, H, W)
        synthesized = denormalize_from_synthesis(synthesized)

        # Step 6: Resize back to original shape
        # synthesized is in (D, H, W) format from the model output
        synthesized_hwd = np.transpose(synthesized, (1, 2, 0))  # (H, W, D)
        synthesized_resized = scipy.ndimage.zoom(
            synthesized_hwd,
            (
                original_shape[0] / synthesized_hwd.shape[0],
                original_shape[1] / synthesized_hwd.shape[1],
                original_shape[2] / synthesized_hwd.shape[2],
            ),
            order=1,
        ).astype(np.float32)

        # Return in raw-like scale [0, 1] — will be z-score normalized
        # together with the other 3 raw modalities in segmentation preprocessing
        return synthesized_resized

    def _run_diffusion_sampling(self, target_mod: str, cond_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            infer_device = self.device
            diffusion = self.diffusions[target_mod]
            try:
                diffusion.to(infer_device)
                cond_on_device = cond_tensor.to(infer_device)
                gen = diffusion.sample(batch_size=1, condition_tensors=cond_on_device)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_exc:
                if "out of memory" not in str(oom_exc).lower() and "CUDA" not in str(oom_exc):
                    raise
                LOGGER.warning("[syn] GPU OOM during synthesis for %s, falling back to CPU", target_mod)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                diffusion.to("cpu")
                cond_on_device = cond_tensor.to("cpu")
                gen = diffusion.sample(batch_size=1, condition_tensors=cond_on_device)
            finally:
                diffusion.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return gen

    # ------------------------------------------------------------------
    # Legacy: synthesize from already-preprocessed stacked array
    # (kept for backward compatibility but NOT recommended)
    # ------------------------------------------------------------------

    def synthesize(self, stacked: np.ndarray, missing_flags: Sequence[int], num_steps: int = 50) -> np.ndarray:
        if stacked.ndim != 4 or stacked.shape[0] != 4:
            raise ValueError(f"Expected stacked shape (4, H, W, D), got {stacked.shape}")
        completed_full = stacked.copy().astype(np.float32, copy=False)
        missing_mods = [MODALITY_ORDER[i] for i, flag in enumerate(missing_flags) if flag]
        LOGGER.info("[syn] Requested synthesis for missing modalities: %s", missing_mods)
        if not missing_mods:
            self.status = "skipped"
            LOGGER.info("[syn] No missing modalities. Nothing to synthesize.")
            return completed_full
        if not self.models:
            LOGGER.warning("[syn] No diffusion models available. Falling back to mean fill.")
            for i, is_missing in enumerate(missing_flags):
                if is_missing:
                    completed_full = self._fallback_fill(completed_full, missing_flags, i)
            self.status = "fallback_mean"
            return completed_full
        any_success = False
        for i, is_missing in enumerate(missing_flags):
            if not is_missing:
                continue
            target_mod = MODALITY_ORDER[i]
            LOGGER.info("[syn] Synthesizing target=%s", target_mod)
            if target_mod not in self.diffusions:
                LOGGER.warning("[syn] No model for target=%s. Using fallback mean.", target_mod)
                completed_full = self._fallback_fill(completed_full, missing_flags, i)
                continue
            try:
                synthesized = self._synthesize_one(completed_full, missing_flags, i, target_mod)
                completed_full[i] = synthesized
                any_success = True
                LOGGER.info("[syn] Completed target=%s", target_mod)
            except Exception as exc:
                LOGGER.exception("[syn] Failed generating target=%s. Using fallback mean.", target_mod)
                self.error = f"Synthesis failed for {target_mod}: {exc}"
                completed_full = self._fallback_fill(completed_full, missing_flags, i)
        self.status = "success" if any_success else "fallback_mean"
        return completed_full

    def _synthesize_one(
        self,
        completed_full: np.ndarray,
        missing_flags: Sequence[int],
        target_idx: int,
        target_mod: str,
    ) -> np.ndarray:
        from synthesis_preprocess import postprocess_synthesized_to_zscore
        cond_mods = self.MODEL_CONFIG[target_mod][1]
        cond_indices = [MODALITY_ORDER.index(m) for m in cond_mods]
        cond_volumes = completed_full[cond_indices]  # (3, H, W, D)
        # Resize to model input size
        target_shape = (SYNTHESIS_IMAGE_SIZE, SYNTHESIS_IMAGE_SIZE, SYNTHESIS_DEPTH_SIZE)
        cond_resized = np.stack(
            [
                scipy.ndimage.zoom(
                    v,
                    (
                        target_shape[0] / v.shape[0],
                        target_shape[1] / v.shape[1],
                        target_shape[2] / v.shape[2],
                    ),
                    order=1,
                )
                for v in cond_volumes
            ],
            axis=0,
        )
        # Normalize each channel independently (percentile → [-1, 1])
        cond_resized = normalize_for_synthesis(cond_resized)
        # (3, H, W, D) → (1, 3, D, H, W)
        cond_tensor = (
            torch.from_numpy(cond_resized)
            .float()
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
        )
        # Run diffusion sampling
        gen = self._run_diffusion_sampling(target_mod, cond_tensor)
        # Post-process: [-1,1] → [0,1] → resize → z-score
        synthesized = gen[0, 0].detach().cpu().numpy()  # (D, H, W)
        synthesized = denormalize_from_synthesis(synthesized)
        original_shape = completed_full.shape[1:]  # (H, W, D)
        synthesized_hwd = np.transpose(synthesized, (1, 2, 0))  # (H, W, D)
        synthesized_resized = scipy.ndimage.zoom(
            synthesized_hwd,
            (
                original_shape[0] / synthesized_hwd.shape[0],
                original_shape[1] / synthesized_hwd.shape[1],
                original_shape[2] / synthesized_hwd.shape[2],
            ),
            order=1,
        ).astype(np.float32)
        # Convert to z-score space to match the rest of stacked
        return postprocess_synthesized_to_zscore(synthesized_resized)

    # Fallback
    def _fallback_fill(self, completed_full: np.ndarray, missing_flags: Sequence[int], target_index: int) -> np.ndarray:
        """Fill a missing modality with the mean of available modalities."""
        avail_vols = [completed_full[j] for j in range(4) if not missing_flags[j] and j != target_index]
        if avail_vols:
            fill = np.mean(avail_vols, axis=0).astype(np.float32, copy=False)
        else:
            fill = np.zeros_like(completed_full[0], dtype=np.float32)
        completed_full[target_index] = fill
        return completed_full

    def _fallback_fill_raw(
        self,
        raw_vols: Dict[str, "np.ndarray | None"],
        target_mod: str,
    ) -> np.ndarray:
        """Fill a missing modality with the mean of available raw modalities."""
        avail = [raw_vols[m] for m in MODALITY_ORDER if raw_vols.get(m) is not None and m != target_mod]
        if avail:
            return np.mean(avail, axis=0).astype(np.float32)
        # All missing — return zeros with shape of first available or a default
        any_vol = next((raw_vols[m] for m in MODALITY_ORDER if raw_vols.get(m) is not None), None)
        if any_vol is not None:
            return np.zeros_like(any_vol, dtype=np.float32)
        return np.zeros((64, 64, 64), dtype=np.float32)
