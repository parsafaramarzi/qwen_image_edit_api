#!/usr/bin/env python3
"""
Memory-efficient loader and inference wrapper for the Qwen Image Edit model.

The previous implementation crashed on machines with limited system RAM
(e.g. 32 GB) because it:

  1. Called ``from_pretrained(...)`` WITHOUT ``torch_dtype``, so the shards were
     first materialised in float32 (~110 GB for this model family) before being
     converted to bfloat16 with ``.to(torch.bfloat16)``. That fp32 spike is what
     pinned RAM to 99% and killed the process.
  2. Called ``.to("cuda")`` to push the entire ~55 GB (bf16) model onto the GPU
     at once, which a 24 GB card (RTX 3090) cannot hold.

This module fixes both problems:

  * Weights are loaded directly in the target dtype (``torch_dtype=bfloat16``)
    with ``low_cpu_mem_usage=True`` so there is never an fp32 spike.
  * By default the transformer and text encoder are loaded in **4-bit** (nf4)
    via bitsandbytes, which brings the resident footprint to ~14-16 GB — small
    enough to fit a 3090 with room for activations.
  * ``enable_model_cpu_offload()`` streams components to the GPU only while they
    are in use, keeping peak VRAM low.

Precision is configurable through the ``QIE_PRECISION`` environment variable:

  * ``4bit``  (default) — nf4 quantized, fits 32 GB RAM + 24 GB VRAM.
  * ``8bit``  — int8 quantized, needs more VRAM/RAM but higher fidelity.
  * ``bf16``  — full bfloat16, only for large-VRAM/large-RAM machines.
"""

from __future__ import annotations

import gc
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from PIL import Image

try:
    from diffusers import QwenImageEditPipeline
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "❌ diffusers is not installed. Run: pip install -r requirements.txt"
    ) from exc


# Default model. Override with QIE_MODEL_ID. The newer "Qwen/Qwen-Image-Edit-2509"
# is a drop-in replacement if you want the latest revision.
DEFAULT_MODEL_ID = os.environ.get("QIE_MODEL_ID", "Qwen/Qwen-Image-Edit")
DEFAULT_PRECISION = os.environ.get("QIE_PRECISION", "4bit").lower()


@dataclass
class LoadState:
    """Tracks pipeline loading progress so the API can report status."""

    status: str = "idle"            # idle | loading | ready | error
    message: str = "Model not loaded yet."
    device: str = "cpu"
    precision: str = DEFAULT_PRECISION
    model_id: str = DEFAULT_MODEL_ID
    error: Optional[str] = None
    started_at: Optional[float] = None
    ready_at: Optional[float] = None

    def as_dict(self) -> dict:
        elapsed = None
        if self.started_at is not None:
            end = self.ready_at or time.time()
            elapsed = round(end - self.started_at, 1)
        return {
            "status": self.status,
            "message": self.message,
            "device": self.device,
            "precision": self.precision,
            "model_id": self.model_id,
            "error": self.error,
            "load_seconds": elapsed,
        }


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_quant_config(precision: str):
    """Return a diffusers PipelineQuantizationConfig, or None for bf16."""
    if precision not in ("4bit", "8bit"):
        return None

    try:
        from diffusers.quantizers import PipelineQuantizationConfig
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "This diffusers version does not support PipelineQuantizationConfig. "
            "Upgrade with: pip install -U diffusers"
        ) from exc

    # bitsandbytes is required for the quantized paths.
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "bitsandbytes is required for 4bit/8bit precision but is not installed. "
            "Install it (pip install bitsandbytes) or set QIE_PRECISION=bf16."
        ) from exc

    if precision == "4bit":
        quant_kwargs = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        }
        backend = "bitsandbytes_4bit"
    else:  # 8bit
        quant_kwargs = {"load_in_8bit": True}
        backend = "bitsandbytes_8bit"

    return PipelineQuantizationConfig(
        quant_backend=backend,
        quant_kwargs=quant_kwargs,
        # The transformer and text encoder are the memory hogs; the VAE is tiny
        # and quantizing it hurts quality, so leave it in bf16.
        components_to_quantize=["transformer", "text_encoder"],
    )


class ModelManager:
    """Owns the pipeline lifecycle and serialises inference calls."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        precision: str = DEFAULT_PRECISION,
    ) -> None:
        self.state = LoadState(model_id=model_id, precision=precision)
        self._pipeline: Optional[QwenImageEditPipeline] = None
        # Only one inference at a time — the model is a single shared GPU resource.
        self._infer_lock = threading.Lock()
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Load the pipeline. Safe to call once; subsequent calls are no-ops."""
        with self._load_lock:
            if self.state.status == "ready":
                return
            self.state.status = "loading"
            self.state.error = None
            self.state.started_at = time.time()
            self.state.message = f"Loading {self.state.model_id} ({self.state.precision})..."
            print(f"📥 {self.state.message}")

            try:
                device = _select_device()
                self.state.device = device

                load_kwargs = {
                    "torch_dtype": torch.bfloat16,   # never load fp32 then convert
                    "low_cpu_mem_usage": True,       # stream shards, no fp32 spike
                }

                quant_config = _build_quant_config(self.state.precision)
                if quant_config is not None:
                    load_kwargs["quantization_config"] = quant_config

                pipeline = QwenImageEditPipeline.from_pretrained(
                    self.state.model_id, **load_kwargs
                )

                if device == "cuda":
                    # Stream components to GPU only while in use → low peak VRAM.
                    # Works with quantized weights and keeps a 3090 within budget.
                    pipeline.enable_model_cpu_offload()
                elif device == "mps":
                    pipeline.to("mps")
                else:
                    print("⚠️  No GPU detected — running on CPU will be very slow.")
                    pipeline.to("cpu")

                # Optional VAE memory savers (harmless, help on tight VRAM).
                try:
                    pipeline.vae.enable_slicing()
                    pipeline.vae.enable_tiling()
                except Exception:
                    pass

                pipeline.set_progress_bar_config(disable=None)

                self._pipeline = pipeline
                self.state.status = "ready"
                self.state.ready_at = time.time()
                self.state.message = (
                    f"Ready on {device} ({self.state.precision}), "
                    f"loaded in {self.state.as_dict()['load_seconds']}s."
                )
                print(f"✅ {self.state.message}")

            except Exception as exc:  # noqa: BLE001
                import traceback

                self.state.status = "error"
                self.state.error = str(exc)
                self.state.message = f"Failed to load model: {exc}"
                print(f"❌ {self.state.message}")
                print(traceback.format_exc())
                raise

    def load_in_background(self) -> None:
        threading.Thread(target=self._safe_background_load, daemon=True).start()

    def _safe_background_load(self) -> None:
        try:
            self.load()
        except Exception:
            # State already records the error; don't crash the server thread.
            pass

    @property
    def is_ready(self) -> bool:
        return self.state.status == "ready" and self._pipeline is not None

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def edit(
        self,
        image: Image.Image,
        prompt: str,
        *,
        num_inference_steps: int = 30,
        true_cfg_scale: float = 4.0,
        negative_prompt: str = "",
        seed: int = 0,
    ) -> Image.Image:
        """Run one image edit. Blocks if another edit is in progress."""
        if not self.is_ready:
            raise RuntimeError(
                f"Model is not ready (status={self.state.status}). "
                f"{self.state.message}"
            )

        image = image.convert("RGB")
        gen_device = "cuda" if self.state.device == "cuda" else "cpu"

        inputs = {
            "image": image,
            "prompt": prompt,
            "generator": torch.Generator(device=gen_device).manual_seed(int(seed)),
            "true_cfg_scale": float(true_cfg_scale),
            "negative_prompt": negative_prompt or " ",
            "num_inference_steps": int(num_inference_steps),
        }

        with self._infer_lock:
            with torch.inference_mode():
                output = self._pipeline(**inputs)
            result = output.images[0]

        # Reclaim any transient VRAM held between requests.
        if self.state.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return result
