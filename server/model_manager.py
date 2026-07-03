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
import inspect
import os
import threading
import time
from dataclasses import dataclass
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

# For non-Qwen models (FLUX Kontext, SD/SDXL img2img, InstructPix2Pix, ...) we
# let diffusers auto-detect the right image-to-image pipeline class.
try:
    from diffusers import AutoPipelineForImage2Image
except ImportError:  # pragma: no cover - older diffusers
    AutoPipelineForImage2Image = None


def _is_qwen_edit(model_id: str) -> bool:
    return "qwen-image-edit" in model_id.lower() or "qwen_image_edit" in model_id.lower()


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


def _build_quant_config(precision: str, components: Optional[list] = None):
    """Return a diffusers PipelineQuantizationConfig, or None for bf16.

    ``components`` restricts which submodules are quantized (e.g. the Qwen
    transformer + text encoder). Pass None to quantize all quantizable modules,
    which is safer for non-Qwen architectures whose component names differ.
    """
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

    kwargs = {"quant_backend": backend, "quant_kwargs": quant_kwargs}
    if components:
        # Restrict to the given components (e.g. Qwen transformer + text encoder);
        # leave the small VAE in bf16 since quantizing it hurts quality.
        kwargs["components_to_quantize"] = components
    return PipelineQuantizationConfig(**kwargs)


# GGUF defaults. Q6_K is the sweet spot for a 24 GB GPU: ~17 GB, near-full
# quality, fully resident (fast). Change the quant with QIE_GGUF_QUANT
# (e.g. Q8_0, Q5_K_M, Q4_K_M), or pin an exact file with QIE_GGUF_FILE.
# Empty by default → the repo is auto-detected from the model id (see
# _resolve_gguf_repo). Set QIE_GGUF_REPO to pin a specific GGUF re-uploader.
DEFAULT_GGUF_REPO = os.environ.get("QIE_GGUF_REPO", "").strip()
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


def _resolve_gguf_repo(model_id: str, explicit: str) -> str:
    """Find a GGUF repo for the model. Uses ``explicit`` if given, else probes
    common community re-uploaders and returns the first that exists."""
    if explicit:
        return explicit

    from huggingface_hub import HfApi

    api = HfApi()
    name = model_id.split("/")[-1]
    candidates = [
        f"QuantStack/{name}-GGUF",
        f"unsloth/{name}-GGUF",
        f"city96/{name}-gguf",
        f"QuantStack/{name}-gguf",
    ]
    for cand in candidates:
        try:
            if api.repo_exists(cand):
                print(f"🔎 Found GGUF repo: {cand}")
                return cand
        except Exception:
            continue
    raise RuntimeError(
        f"No GGUF repo found for '{model_id}' (tried {', '.join(candidates)}). "
        "Set gguf_repo explicitly, or use precision 4bit/8bit/bf16 instead."
    )


def _build_gguf_pipeline(model_id: str, repo: str, file: str, quant: str, base: str):
    """Load a GGUF-quantized transformer + the rest of the pipeline from HF.

    Currently GGUF is supported only for Qwen-Image-Edit models. Requires a
    recent diffusers (the 2511 GGUF loader fix landed in huggingface/diffusers
    PR #12894) and the ``gguf`` package.
    """
    if not _is_qwen_edit(model_id) and not _is_qwen_edit(base):
        raise RuntimeError(
            "GGUF precision currently supports only Qwen-Image-Edit models. "
            "For other models pick precision 4bit/8bit/bf16."
        )

    repo = _resolve_gguf_repo(model_id, repo)

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
    base_model = base or model_id
    if base_model.lower() in ("qwen/qwen-image-edit", ""):
        base_model = "Qwen/Qwen-Image-Edit-2511"

    filename = _resolve_gguf_filename(repo, file, quant)
    print(f"📥 Fetching GGUF transformer {repo}/{filename} ...")
    gguf_path = hf_hub_download(repo_id=repo, filename=filename)

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
    return pipeline, base_model


# Directory where downloaded LoRA files are cached.
LORA_CACHE_DIR = os.path.expanduser(os.environ.get("QIE_LORA_DIR", "~/.cache/qwen_loras"))


def _download_lora(url: str) -> tuple:
    """Download a LoRA file (e.g. from Civitai) to the local cache.

    Returns (directory, filename) suitable for pipeline.load_lora_weights.
    """
    import re
    import requests

    os.makedirs(LORA_CACHE_DIR, exist_ok=True)

    # A Civitai API token can be appended if the model requires auth.
    token = os.environ.get("QIE_CIVITAI_TOKEN", "").strip()
    if token and "civitai.com" in url and "token=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}token={token}"

    with requests.get(url, stream=True, timeout=120, allow_redirects=True) as resp:
        resp.raise_for_status()
        # Prefer the server-provided filename; fall back to the URL tail.
        name = None
        cd = resp.headers.get("content-disposition", "")
        m = re.search(r'filename="?([^"\r\n;]+)"?', cd)
        if m:
            name = m.group(1)
        if not name:
            name = url.split("?")[0].rstrip("/").split("/")[-1] or "lora.safetensors"
        if not name.endswith((".safetensors", ".bin", ".pt")):
            name += ".safetensors"

        dest = os.path.join(LORA_CACHE_DIR, name)
        if not os.path.exists(dest):  # simple cache: skip re-download
            print(f"📥 Downloading LoRA → {dest}")
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
    return LORA_CACHE_DIR, name


def _resolve_lora(source: str) -> tuple:
    """Return (path_or_repo, weight_name) for pipeline.load_lora_weights."""
    source = source.strip()
    if source.startswith("http://") or source.startswith("https://"):
        return _download_lora(source)
    # Otherwise treat as an HF repo id (optionally "repo::weight_name.safetensors").
    if "::" in source:
        repo, weight = source.split("::", 1)
        return repo.strip(), weight.strip()
    return source, None


class ModelManager:
    """Owns the pipeline lifecycle and serialises inference calls.

    Supports switching models at runtime via ``reload()`` and loading arbitrary
    diffusers image-to-image models (not just Qwen) via
    ``AutoPipelineForImage2Image``.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        precision: str = DEFAULT_PRECISION,
    ) -> None:
        self.state = LoadState(model_id=model_id, precision=precision)
        self._pipeline = None
        self._is_plus = False   # True for a Qwen 2509/2511 "Plus" pipeline
        self._is_qwen = _is_qwen_edit(model_id)
        # Names accepted by the current pipeline's __call__ (None = accept all).
        self._call_params: Optional[set] = None
        # GGUF settings (overridable per-reload from the client).
        self._gguf_repo = DEFAULT_GGUF_REPO
        self._gguf_quant = DEFAULT_GGUF_QUANT
        self._gguf_file = DEFAULT_GGUF_FILE
        self._gguf_base = os.environ.get("QIE_GGUF_BASE", "")
        # Optional LoRA (URL/repo + strength), applied after the base model.
        self._lora_source = os.environ.get("QIE_LORA", "").strip()
        self._lora_scale = float(os.environ.get("QIE_LORA_SCALE", "1.0"))
        self._lora_active = False
        self._lora_status = "none"   # none | loading | active | failed: ...
        # Live inference progress (polled by the client via /progress).
        self._progress = {"active": False, "stage": "idle", "step": 0, "total": 0}
        # Only one inference at a time — the model is a single shared GPU resource.
        self._infer_lock = threading.Lock()
        self._load_lock = threading.Lock()

    @property
    def progress(self) -> dict:
        return dict(self._progress)

    def lora_info(self) -> dict:
        return {
            "active": self._lora_active,
            "status": self._lora_status,
            "source": self._lora_source,
            "scale": self._lora_scale,
        }

    # ------------------------------------------------------------------ #
    # LoRA (apply/remove on the already-loaded model)
    # ------------------------------------------------------------------ #
    def apply_lora(self, source: str, scale: float) -> None:
        """Download + apply a LoRA to the current pipeline, in the background."""
        source = (source or "").strip()

        def worker():
            with self._load_lock:
                if not self.is_ready:
                    self._lora_status = "failed: no model loaded"
                    return
                self._lora_status = "loading"
                try:
                    path, weight_name = _resolve_lora(source)
                    kwargs = {"adapter_name": "custom"}
                    if weight_name:
                        kwargs["weight_name"] = weight_name
                    with self._infer_lock:
                        try:
                            self._pipeline.unload_lora_weights()
                        except Exception:
                            pass
                        self._pipeline.load_lora_weights(path, **kwargs)
                        try:
                            self._pipeline.set_adapters("custom", float(scale))
                        except Exception:
                            pass
                    self._lora_source = source
                    self._lora_scale = float(scale)
                    self._lora_active = True
                    self._lora_status = "active"
                    print(f"✅ LoRA applied: {source} (scale {scale})")
                except Exception as exc:  # noqa: BLE001
                    self._lora_active = False
                    self._lora_status = f"failed: {exc}"
                    print(f"❌ LoRA failed: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def remove_lora(self) -> None:
        """Remove any applied LoRA from the current pipeline."""
        with self._infer_lock:
            if self._pipeline is not None:
                try:
                    self._pipeline.unload_lora_weights()
                except Exception:
                    pass
        self._lora_active = False
        self._lora_source = ""
        self._lora_status = "none"

    def unload_model(self) -> None:
        """Free the current model + VRAM; server goes back to an idle state."""
        with self._load_lock:
            self._teardown()
            self._lora_active = False
            self._lora_status = "none"
            self.state = LoadState(
                status="idle", message="No model loaded.", model_id="", precision=""
            )

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Load the currently-configured model. No-op if already ready."""
        with self._load_lock:
            if self.state.status == "ready":
                return
            self._do_load()

    def load_in_background(self) -> None:
        threading.Thread(target=self._safe_background_load, daemon=True).start()

    def reload(
        self,
        model_id: str,
        precision: str,
        *,
        gguf_repo: Optional[str] = None,
        gguf_file: Optional[str] = None,
        gguf_quant: Optional[str] = None,
        gguf_base: Optional[str] = None,
        lora_source: Optional[str] = None,
        lora_scale: Optional[float] = None,
    ) -> None:
        """Switch to a different model/precision. Runs in the background;
        poll the state (via /health) until status is 'ready' or 'error'."""

        def worker() -> None:
            with self._load_lock:
                self._teardown()
                self._gguf_repo = (gguf_repo or DEFAULT_GGUF_REPO).strip()
                self._gguf_quant = (gguf_quant or DEFAULT_GGUF_QUANT).strip()
                self._gguf_file = (gguf_file or "").strip()
                self._gguf_base = (gguf_base or "").strip()
                self._lora_source = (lora_source or "").strip()
                self._lora_scale = float(lora_scale) if lora_scale is not None else 1.0
                self.state = LoadState(
                    model_id=model_id.strip(),
                    precision=(precision or DEFAULT_PRECISION).strip().lower(),
                )
                try:
                    self._do_load()
                except Exception:
                    pass  # error already recorded in state

        threading.Thread(target=worker, daemon=True).start()

    def _safe_background_load(self) -> None:
        try:
            self.load()
        except Exception:
            pass  # state already records the error

    def _teardown(self) -> None:
        """Free the current pipeline + VRAM before loading another model."""
        with self._infer_lock:
            self._pipeline = None
            self._call_params = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _do_load(self) -> None:
        """Actually build the pipeline. Assumes the load lock is held."""
        self.state.status = "loading"
        self.state.error = None
        self.state.started_at = time.time()
        self.state.message = f"Loading {self.state.model_id} ({self.state.precision})..."
        print(f"📥 {self.state.message}")

        try:
            device = _select_device()
            self.state.device = device
            self._is_qwen = _is_qwen_edit(self.state.model_id) or (
                self.state.precision == "gguf"
            )

            pipeline = self._construct_pipeline()

            # Apply an optional LoRA before offload hooks are attached.
            lora_note = self._apply_lora(pipeline)

            self._apply_offload(pipeline, device)

            # Optional VAE memory savers (harmless, help on tight VRAM).
            for meth in ("enable_slicing", "enable_tiling"):
                try:
                    getattr(pipeline.vae, meth)()
                except Exception:
                    pass

            try:
                pipeline.set_progress_bar_config(disable=None)
            except Exception:
                pass

            self._pipeline = pipeline
            self._call_params = _accepted_call_params(pipeline)
            self.state.status = "ready"
            self.state.ready_at = time.time()
            self.state.message = (
                f"Ready on {device} ({self.state.precision}), "
                f"loaded in {self.state.as_dict()['load_seconds']}s.{lora_note}"
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

    def _construct_pipeline(self):
        """Build the right pipeline for the configured model + precision."""
        model_id = self.state.model_id
        precision = self.state.precision

        if precision == "gguf":
            pipeline, base = _build_gguf_pipeline(
                model_id, self._gguf_repo, self._gguf_file,
                self._gguf_quant, self._gguf_base,
            )
            self._is_plus = _uses_plus_pipeline(base)
            return pipeline

        load_kwargs = {
            "torch_dtype": torch.bfloat16,   # never load fp32 then convert
            "low_cpu_mem_usage": True,       # stream shards, no fp32 spike
        }

        if _is_qwen_edit(model_id):
            # Qwen edit models: transformer + text_encoder are the memory hogs.
            quant_config = _build_quant_config(precision, ["transformer", "text_encoder"])
            if quant_config is not None:
                load_kwargs["quantization_config"] = quant_config
            self._is_plus = _uses_plus_pipeline(model_id)
            if self._is_plus:
                if QwenImageEditPlusPipeline is None:
                    raise RuntimeError(
                        f"{model_id} needs QwenImageEditPlusPipeline, missing from "
                        "this diffusers. Upgrade: pip install -U diffusers"
                    )
                pipeline_cls = QwenImageEditPlusPipeline
            else:
                pipeline_cls = QwenImageEditPipeline
            return pipeline_cls.from_pretrained(model_id, **load_kwargs)

        # Any other model: auto-detect the image-to-image pipeline class.
        self._is_plus = False
        if AutoPipelineForImage2Image is None:
            raise RuntimeError(
                "AutoPipelineForImage2Image is unavailable; upgrade diffusers."
            )
        # Quantize all quantizable submodules (component names vary across
        # architectures — unet vs transformer — so don't restrict them).
        quant_config = _build_quant_config(precision, None)
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
        return AutoPipelineForImage2Image.from_pretrained(model_id, **load_kwargs)

    def _apply_lora(self, pipeline) -> str:
        """Load an optional LoRA onto the pipeline. Returns a status suffix.

        A failure here does NOT fail the whole load — the base model still
        works; we just note that the LoRA couldn't be applied (common on GGUF).
        """
        if not self._lora_source:
            self._lora_active = False
            self._lora_status = "none"
            return ""
        try:
            path, weight_name = _resolve_lora(self._lora_source)
            kwargs = {"adapter_name": "custom"}
            if weight_name:
                kwargs["weight_name"] = weight_name
            print(f"🎨 Loading LoRA: {self._lora_source} (scale {self._lora_scale})")
            pipeline.load_lora_weights(path, **kwargs)
            try:
                pipeline.set_adapters("custom", self._lora_scale)
            except Exception:
                pass  # some versions apply at load; scale then defaults to 1.0
            self._lora_active = True
            self._lora_status = "active"
            return f" + LoRA (scale {self._lora_scale})"
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  LoRA not applied: {exc}")
            self._lora_active = False
            self._lora_status = f"failed: {exc}"
            return f" [LoRA FAILED: {exc}]"

    def _apply_offload(self, pipeline, device: str) -> None:
        if device == "cuda":
            if self.state.precision in ("bf16", "max"):
                # Full precision larger than VRAM → stream layer-by-layer.
                pipeline.enable_sequential_cpu_offload()
            else:
                # Quantized/GGUF: stream whole components to GPU as used.
                pipeline.enable_model_cpu_offload()
        elif device == "mps":
            pipeline.to("mps")
        else:
            print("⚠️  No GPU detected — running on CPU will be very slow.")
            pipeline.to("cpu")

    @property
    def is_ready(self) -> bool:
        return self.state.status == "ready" and self._pipeline is not None

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def edit(
        self,
        images,
        prompt: str,
        *,
        num_inference_steps: int = 40,
        true_cfg_scale: float = 4.0,
        negative_prompt: str = "",
        seed: int = 0,
        guidance_scale: float = 1.0,
    ) -> Image.Image:
        """Run one image edit. Blocks if another edit is in progress.

        ``images`` may be a single PIL image or a list (the Plus pipeline
        supports up to ~3 reference images; refer to them as "image 1/2/3" in
        the prompt). Optional parameters are filtered to what the loaded
        pipeline accepts, so this also works across FLUX Kontext, SD img2img, etc.
        """
        if not self.is_ready:
            raise RuntimeError(
                f"Model is not ready (status={self.state.status}). "
                f"{self.state.message}"
            )

        if isinstance(images, Image.Image):
            images = [images]
        images = [im.convert("RGB") for im in images]
        gen_device = "cuda" if self.state.device == "cuda" else "cpu"
        steps = int(num_inference_steps)

        # Always-present inputs.
        inputs = {
            # The Plus (2509/2511) pipeline takes a list; classic ones a single
            # image, so hand non-Plus pipelines just the first image.
            "image": images if self._is_plus else images[0],
            "prompt": prompt,
            "generator": torch.Generator(device=gen_device).manual_seed(int(seed)),
        }

        # Optional inputs — included only if the pipeline's __call__ accepts them.
        optional = {
            "num_inference_steps": steps,
            "guidance_scale": float(guidance_scale),
            "true_cfg_scale": float(true_cfg_scale),
        }
        # Qwen wants a single space for "no negative prompt"; others take "".
        if negative_prompt:
            optional["negative_prompt"] = negative_prompt
        elif self._is_qwen:
            optional["negative_prompt"] = " "

        for key, value in optional.items():
            if self._call_params is None or key in self._call_params:
                inputs[key] = value

        # Per-step progress callback (if the pipeline supports the newer API).
        def _on_step(pipe, step, timestep, cbkw):
            self._progress.update(
                {"active": True, "stage": "denoising", "step": step + 1, "total": steps}
            )
            return cbkw

        if self._call_params is None or "callback_on_step_end" in self._call_params:
            inputs["callback_on_step_end"] = _on_step

        with self._infer_lock:
            # "preparing" covers upload received + prompt/image encoding, before
            # the first denoising step fires the callback.
            self._progress.update(
                {"active": True, "stage": "preparing", "step": 0, "total": steps}
            )
            try:
                with torch.inference_mode():
                    output = self._pipeline(**inputs)
                result = output.images[0]
            finally:
                self._progress.update(
                    {"active": False, "stage": "idle", "step": 0, "total": 0}
                )

        if self.state.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return result


def _accepted_call_params(pipeline) -> Optional[set]:
    """Return the set of keyword names the pipeline's __call__ accepts, or None
    if it takes **kwargs (accept anything)."""
    try:
        sig = inspect.signature(pipeline.__call__)
    except (TypeError, ValueError):
        return None
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return None
    return set(sig.parameters)


# ---------------------------------------------------------------------------- #
# Hugging Face cache management (downloaded models)
# ---------------------------------------------------------------------------- #
def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def list_cached_models() -> list:
    """Return the models already downloaded to the HF cache (id + size)."""
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
    except Exception as exc:  # noqa: BLE001
        return []
    out = []
    for repo in info.repos:
        if getattr(repo, "repo_type", "model") != "model":
            continue
        out.append(
            {
                "repo_id": repo.repo_id,
                "size": repo.size_on_disk,
                "size_str": _human_size(repo.size_on_disk),
            }
        )
    return sorted(out, key=lambda r: -r["size"])


def cached_repo_ids() -> set:
    return {m["repo_id"] for m in list_cached_models()}


def delete_cached_model(repo_id: str) -> dict:
    """Delete all cached revisions of a model repo. Returns freed-space info."""
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir()
    revisions = [
        rev.commit_hash
        for repo in info.repos
        if repo.repo_id == repo_id
        for rev in repo.revisions
    ]
    if not revisions:
        raise RuntimeError(f"'{repo_id}' is not in the cache.")
    strategy = info.delete_revisions(*revisions)
    freed = strategy.expected_freed_size
    strategy.execute()
    return {"repo_id": repo_id, "freed": freed, "freed_str": _human_size(freed)}


def validate_model(model_id: str) -> dict:
    """Check whether a model repo exists on HF and whether it's already cached."""
    model_id = model_id.strip()
    result = {"model_id": model_id, "exists": False, "cached": False, "message": ""}
    if not model_id:
        result["message"] = "Empty model id."
        return result

    result["cached"] = model_id in cached_repo_ids()
    try:
        from huggingface_hub import HfApi

        exists = HfApi().repo_exists(model_id)
        result["exists"] = bool(exists)
        result["message"] = "Found on Hugging Face." if exists else "Not found on Hugging Face."
    except Exception as exc:  # noqa: BLE001
        # If we can't reach HF but it's cached, it's still usable offline.
        result["message"] = (
            "Already downloaded (offline)." if result["cached"]
            else f"Could not verify: {exc}"
        )
        result["exists"] = result["cached"]
    return result
