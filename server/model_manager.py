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

  * ``4bit``  (default) — nf4 quantized, smallest/fastest, fits 32 GB RAM +
    24 GB VRAM comfortably. Some quality loss (the transformer is quant-
    sensitive), so edits can look softer/noisier.
  * ``8bit``  — int8 quantized. Higher fidelity, but the transformer alone is
    ~20 GB so it does NOT fit a 24 GB card alongside the text encoder.
  * ``gguf`` — GGUF K-quant transformer (default Q6_K, ~17 GB) loaded from a
    single .gguf file, rest of the pipeline in bf16. Higher quality-per-bit than
    bnb 4-bit AND fully GPU-resident, so it's near-full quality at full speed —
    the sweet spot for a 24 GB card. Needs a recent diffusers + the ``gguf``
    package. Pick the quant via QIE_GGUF_FILE / QIE_GGUF_REPO.
  * ``max`` / ``bf16`` — FULL precision, no quantization. On a small GPU this
    uses sequential CPU offload: the complete bf16 weights live in system RAM
    (~57 GB, needs a 64 GB box) and stream through the GPU layer-by-layer.
    Highest possible quality; slowest per image (weights cross the PCIe bus
    every step). Use ``gguf`` instead unless you need bit-exact full precision.
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

# The 2509/2511 revisions use a DIFFERENT pipeline class (the "Plus" pipeline,
# with multi-image + better realism/consistency). It only exists in newer
# diffusers, so import it defensively.
try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError:  # pragma: no cover - older diffusers
    QwenImageEditPlusPipeline = None


# Default model. Override with QIE_MODEL_ID. Qwen-Image-Edit-2511 is the latest
# revision (improved realism, character consistency, less drift) and uses the
# "Plus" pipeline, which is auto-detected below.
DEFAULT_MODEL_ID = os.environ.get("QIE_MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
DEFAULT_PRECISION = os.environ.get("QIE_PRECISION", "4bit").lower()


def _uses_plus_pipeline(model_id: str) -> bool:
    """The 2509/2511 (and any '...-Plus') revisions need QwenImageEditPlusPipeline."""
    mid = model_id.lower()
    return any(tag in mid for tag in ("2509", "2511", "plus"))


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


# GGUF defaults. Q6_K is the sweet spot for a 24 GB GPU: ~17 GB, near-full
# quality, fully resident (fast). Change the quant with QIE_GGUF_QUANT
# (e.g. Q8_0, Q5_K_M, Q4_K_M), or pin an exact file with QIE_GGUF_FILE.
DEFAULT_GGUF_REPO = os.environ.get(
    "QIE_GGUF_REPO", "QuantStack/Qwen-Image-Edit-2511-GGUF"
)
DEFAULT_GGUF_QUANT = os.environ.get("QIE_GGUF_QUANT", "Q6_K")
# If set, use this exact filename; otherwise auto-discover by quant tag.
DEFAULT_GGUF_FILE = os.environ.get("QIE_GGUF_FILE", "").strip()


def _resolve_gguf_filename(repo: str, explicit: str, quant: str) -> str:
    """Return the .gguf filename to download.

    Naming across GGUF repos is inconsistent (hyphens vs underscores, casing),
    so unless an exact file is pinned we list the repo and match by quant tag.
    """
    if explicit:
        return explicit

    from huggingface_hub import list_repo_files

    files = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    if not files:
        raise RuntimeError(f"No .gguf files found in repo '{repo}'.")

    tag = quant.lower().replace("-", "_")
    matches = [f for f in files if tag in f.lower().replace("-", "_")]
    if not matches:
        raise RuntimeError(
            f"No GGUF file matching quant '{quant}' in '{repo}'. "
            f"Available: {', '.join(sorted(files))}. "
            f"Set QIE_GGUF_QUANT to one of these or QIE_GGUF_FILE to an exact name."
        )
    # Prefer the shortest match (avoids picking multi-part split files if a
    # single-file variant exists).
    return sorted(matches, key=len)[0]


def _build_gguf_pipeline(model_id: str):
    """Load a GGUF-quantized transformer + the rest of the pipeline from HF.

    Requires a recent diffusers (the 2511 GGUF loader fix landed in
    huggingface/diffusers PR #12894 — a released version may be too old, in
    which case install from git) and the ``gguf`` package.
    """
    try:
        from diffusers import GGUFQuantizationConfig, QwenImageTransformer2DModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "GGUF loading needs a recent diffusers with GGUFQuantizationConfig. "
            "Install the latest: pip install -U 'diffusers[gguf]' gguf  "
            "(or from git: pip install git+https://github.com/huggingface/diffusers)"
        ) from exc

    if QwenImageEditPlusPipeline is None:
        raise RuntimeError(
            "GGUF mode targets the 2511 'Plus' pipeline, which this diffusers "
            "version lacks. Upgrade: pip install -U diffusers"
        )

    from huggingface_hub import hf_hub_download

    # The GGUF file only holds the transformer; the text encoder, VAE, tokenizer
    # and config come from the base HF repo.
    base_model = os.environ.get("QIE_GGUF_BASE", model_id)
    if base_model.lower() in ("qwen/qwen-image-edit", ""):
        # A GGUF repo is 2511/2509-based; default the base to 2511 so configs match.
        base_model = "Qwen/Qwen-Image-Edit-2511"

    filename = _resolve_gguf_filename(
        DEFAULT_GGUF_REPO, DEFAULT_GGUF_FILE, DEFAULT_GGUF_QUANT
    )
    print(f"📥 Fetching GGUF transformer {DEFAULT_GGUF_REPO}/{filename} ...")
    gguf_path = hf_hub_download(repo_id=DEFAULT_GGUF_REPO, filename=filename)

    print(f"🔧 Loading GGUF transformer (base config: {base_model}) ...")
    transformer = QwenImageTransformer2DModel.from_single_file(
        gguf_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        config=base_model,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        base_model,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    return pipeline


class ModelManager:
    """Owns the pipeline lifecycle and serialises inference calls."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        precision: str = DEFAULT_PRECISION,
    ) -> None:
        self.state = LoadState(model_id=model_id, precision=precision)
        self._pipeline = None
        self._is_plus = False  # True when running a 2509/2511 "Plus" pipeline
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

                if self.state.precision == "gguf":
                    # GGUF: load a K-quant transformer from a single .gguf file
                    # (higher quality-per-bit than bnb 4-bit) and pull the rest
                    # of the pipeline from the HF repo. Best quality/speed on a
                    # 24 GB GPU — near full-precision, but fully resident (fast).
                    pipeline = _build_gguf_pipeline(self.state.model_id)
                    self._is_plus = _uses_plus_pipeline(
                        os.environ.get("QIE_GGUF_BASE", self.state.model_id)
                    )
                else:
                    load_kwargs = {
                        "torch_dtype": torch.bfloat16,   # never load fp32 then convert
                        "low_cpu_mem_usage": True,       # stream shards, no fp32 spike
                    }

                    quant_config = _build_quant_config(self.state.precision)
                    if quant_config is not None:
                        load_kwargs["quantization_config"] = quant_config

                    # Pick the right pipeline class for the model revision.
                    self._is_plus = _uses_plus_pipeline(self.state.model_id)
                    if self._is_plus:
                        if QwenImageEditPlusPipeline is None:
                            raise RuntimeError(
                                f"{self.state.model_id} needs QwenImageEditPlusPipeline, "
                                "which this diffusers version lacks. Upgrade with: "
                                "pip install -U diffusers"
                            )
                        pipeline_cls = QwenImageEditPlusPipeline
                    else:
                        pipeline_cls = QwenImageEditPipeline

                    pipeline = pipeline_cls.from_pretrained(
                        self.state.model_id, **load_kwargs
                    )

                if device == "cuda":
                    if self.state.precision in ("bf16", "max"):
                        # Full-precision weights (~57 GB) are far larger than a
                        # 24 GB GPU, so stream them layer-by-layer from system
                        # RAM. This runs the UNQUANTIZED model at maximum quality
                        # on a small GPU — the trade-off is speed (weights cross
                        # the PCIe bus every step). Needs enough system RAM to
                        # hold the full model (~57 GB → 64 GB box is the minimum).
                        pipeline.enable_sequential_cpu_offload()
                    else:
                        # Quantized (4bit/8bit): components fit, so stream whole
                        # models to the GPU only while in use → low peak VRAM.
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
        num_inference_steps: int = 40,
        true_cfg_scale: float = 4.0,
        negative_prompt: str = "",
        seed: int = 0,
        guidance_scale: float = 1.0,
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
            # The Plus (2509/2511) pipeline expects a list of images; the classic
            # pipeline expects a single image.
            "image": [image] if self._is_plus else image,
            "prompt": prompt,
            "generator": torch.Generator(device=gen_device).manual_seed(int(seed)),
            "true_cfg_scale": float(true_cfg_scale),
            "negative_prompt": negative_prompt or " ",
            "num_inference_steps": int(num_inference_steps),
            # Distilled guidance embedding (separate from true_cfg_scale). Official
            # Qwen examples use 1.0; higher values tend to over-saturate.
            "guidance_scale": float(guidance_scale),
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
