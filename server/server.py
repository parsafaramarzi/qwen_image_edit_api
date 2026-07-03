#!/usr/bin/env python3
"""
Qwen Image Edit - FastAPI inference server.

Runs the heavy model on your GPU box and exposes a small HTTP API so a
lightweight client (e.g. the Tkinter GUI in ../client) can drive it from
another machine.

Endpoints
---------
GET  /health            Liveness + model status (JSON).
GET  /status            Alias of /health.
GET  /models            Suggested models the client can load.
POST /load              JSON: switch to a different model/precision (async).
POST /edit              multipart/form-data: image file + edit parameters,
                        returns the edited image as PNG bytes.

Run it
------
    python server.py
    # or
    uvicorn server:app --host 0.0.0.0 --port 8000

The loaded model can also be switched at runtime by the client via POST /load,
so these just set the initial model.

Environment
-----------
    QIE_MODEL_ID   HF repo id (default: Qwen/Qwen-Image-Edit-2511)
    QIE_PRECISION  gguf | 4bit | 8bit | max/bf16  (default: 4bit)
    QIE_GGUF_REPO/FILE/QUANT/BASE   GGUF source overrides
    QIE_HOST       bind host (default: 0.0.0.0)
    QIE_PORT       bind port (default: 8000)
    QIE_API_KEY    if set, clients must send it as the X-API-Key header
"""

from __future__ import annotations

import io
import os

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from PIL import Image

from model_manager import (
    DEFAULT_MODEL_ID,
    DEFAULT_PRECISION,
    ModelManager,
    delete_cached_model,
    list_cached_models,
    validate_model,
)

# Suggested models for the client's dropdown. The first is the default.
SUGGESTED_MODELS = [
    {"model_id": "Qwen/Qwen-Image-Edit-2511", "precision": "gguf",
     "label": "Qwen Image Edit 2511 (GGUF Q6_K) — recommended"},
    {"model_id": "Qwen/Qwen-Image-Edit-2511", "precision": "max",
     "label": "Qwen Image Edit 2511 (full bf16, slow)"},
    {"model_id": "Qwen/Qwen-Image-Edit-2509", "precision": "4bit",
     "label": "Qwen Image Edit 2509 (4-bit)"},
    {"model_id": "Qwen/Qwen-Image-Edit", "precision": "4bit",
     "label": "Qwen Image Edit original (4-bit)"},
    {"model_id": "black-forest-labs/FLUX.1-Kontext-dev", "precision": "4bit",
     "label": "FLUX.1 Kontext dev (4-bit) — non-Qwen edit model"},
    {"model_id": "timbrooks/instruct-pix2pix", "precision": "bf16",
     "label": "InstructPix2Pix (bf16) — non-Qwen edit model"},
]

API_KEY = os.environ.get("QIE_API_KEY", "").strip()

app = FastAPI(
    title="Qwen Image Edit API",
    version="1.0.0",
    description="HTTP inference server for the Qwen Image Edit model.",
)

# Allow the desktop client (and browsers) to call from anywhere on the LAN.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ModelManager(model_id=DEFAULT_MODEL_ID, precision=DEFAULT_PRECISION)


@app.on_event("startup")
def _startup() -> None:
    # Boot idle — do NOT preload a model. The client chooses and loads one on
    # demand via POST /load. Set QIE_AUTOLOAD=1 to restore boot-time preload.
    if os.environ.get("QIE_AUTOLOAD", "0").lower() in ("1", "true", "yes"):
        manager.load_in_background()
    else:
        manager.state.status = "idle"
        manager.state.model_id = ""
        manager.state.precision = ""
        manager.state.message = "No model loaded — load one from the client."


def _check_api_key(provided: str | None) -> None:
    if API_KEY and provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@app.get("/health")
@app.get("/status")
def health() -> JSONResponse:
    data = manager.state.as_dict()
    data["lora"] = manager.lora_info()
    data["task"] = manager.task_info()
    return JSONResponse(data)


class LoraRequest(BaseModel):
    lora_source: str
    lora_scale: float = 1.0


@app.post("/lora/load")
def lora_load(req: LoraRequest, x_api_key: str | None = Header(default=None)) -> JSONResponse:
    """Apply a LoRA to the already-loaded model (async; poll /health.lora)."""
    _check_api_key(x_api_key)
    if not manager.is_ready:
        raise HTTPException(status_code=409, detail="Load a model first.")
    if not req.lora_source.strip():
        raise HTTPException(status_code=400, detail="lora_source is required.")
    manager.apply_lora(req.lora_source, req.lora_scale)
    return JSONResponse(manager.lora_info())


@app.post("/lora/unload")
def lora_unload(x_api_key: str | None = Header(default=None)) -> JSONResponse:
    _check_api_key(x_api_key)
    manager.remove_lora()
    return JSONResponse(manager.lora_info())


@app.post("/model/unload")
def model_unload(x_api_key: str | None = Header(default=None)) -> JSONResponse:
    """Free the model + VRAM. Server returns to idle until a model is loaded."""
    _check_api_key(x_api_key)
    manager.unload_model()
    return JSONResponse(manager.state.as_dict())


@app.get("/models")
def models() -> JSONResponse:
    # Annotate each suggestion with whether it's already downloaded.
    cached = {m["repo_id"] for m in list_cached_models()}
    items = [dict(m, cached=(m["model_id"] in cached)) for m in SUGGESTED_MODELS]
    return JSONResponse({"models": items})


@app.get("/progress")
def progress() -> JSONResponse:
    return JSONResponse(manager.progress)


@app.get("/cache")
def cache() -> JSONResponse:
    """List models already downloaded to the HF cache (with sizes)."""
    return JSONResponse({"models": list_cached_models()})


class DeleteRequest(BaseModel):
    repo_id: str


@app.post("/cache/delete")
def cache_delete(req: DeleteRequest, x_api_key: str | None = Header(default=None)) -> JSONResponse:
    _check_api_key(x_api_key)
    if manager.state.model_id == req.repo_id and manager.is_ready:
        raise HTTPException(
            status_code=409,
            detail="That model is currently loaded. Load a different model first.",
        )
    try:
        info = delete_cached_model(req.repo_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(info)


class ValidateRequest(BaseModel):
    model_id: str


@app.post("/validate")
def validate(req: ValidateRequest, x_api_key: str | None = Header(default=None)) -> JSONResponse:
    _check_api_key(x_api_key)
    return JSONResponse(validate_model(req.model_id))


class LoadRequest(BaseModel):
    model_id: str
    precision: str = "gguf"
    # GGUF-only overrides (ignored for other precisions):
    gguf_repo: str | None = None
    gguf_file: str | None = None
    gguf_quant: str | None = None
    gguf_base: str | None = None
    # Optional LoRA: HF repo id ("repo::weight.safetensors") or a direct URL
    # (e.g. a Civitai download link). Empty = no LoRA.
    lora_source: str | None = None
    lora_scale: float | None = None


@app.post("/load")
def load_model(req: LoadRequest, x_api_key: str | None = Header(default=None)) -> JSONResponse:
    """Switch the loaded model. Returns immediately with status='loading';
    the client should poll /health until status is 'ready' or 'error'."""
    _check_api_key(x_api_key)
    if not req.model_id.strip():
        raise HTTPException(status_code=400, detail="model_id is required.")
    manager.reload(
        req.model_id,
        req.precision,
        gguf_repo=req.gguf_repo,
        gguf_file=req.gguf_file,
        gguf_quant=req.gguf_quant,
        gguf_base=req.gguf_base,
        lora_source=req.lora_source,
        lora_scale=req.lora_scale,
    )
    return JSONResponse(manager.state.as_dict())


@app.post("/edit")
@app.post("/generate")
async def edit(
    image: list[UploadFile] | None = File(None, description="Input image(s) — 0 to 3."),
    prompt: str = Form(..., description="Prompt / edit instruction."),
    num_inference_steps: int = Form(40),
    true_cfg_scale: float = Form(4.0),
    negative_prompt: str = Form(""),
    seed: int = Form(0),
    guidance_scale: float = Form(1.0),
    width: int = Form(0),
    height: int = Form(0),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)

    if not manager.is_ready:
        # 503: model still warming up or failed to load.
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Model not ready.",
                "state": manager.state.as_dict(),
            },
        )

    # 0-3 images under the repeated "image" field. None/empty ⇒ generation.
    uploads = image or []
    if not isinstance(uploads, list):
        uploads = [uploads]
    try:
        pil_images = []
        for up in uploads:
            raw = await up.read()
            if raw:
                pil_images.append(Image.open(io.BytesIO(raw)))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    try:
        result = manager.run(
            pil_images,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            seed=seed,
            guidance_scale=guidance_scale,
            width=width or None,
            height=height or None,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


def main() -> None:
    import uvicorn

    host = os.environ.get("QIE_HOST", "0.0.0.0")
    port = int(os.environ.get("QIE_PORT", "8000"))

    # Optional TLS: set QIE_SSL_CERT + QIE_SSL_KEY to encrypt the connection so
    # traffic can't be intercepted. A self-signed cert is fine for personal use
    # (generate with: openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem
    #  -out cert.pem -days 365 -subj "/CN=<server-ip>").
    ssl_cert = os.environ.get("QIE_SSL_CERT", "").strip()
    ssl_key = os.environ.get("QIE_SSL_KEY", "").strip()
    ssl_kwargs = {}
    scheme = "http"
    if ssl_cert and ssl_key:
        ssl_kwargs = {"ssl_certfile": ssl_cert, "ssl_keyfile": ssl_key}
        scheme = "https"

    autoload = os.environ.get("QIE_AUTOLOAD", "0").lower() in ("1", "true", "yes")
    print(f"🚀 Starting Qwen Image Edit API on {scheme}://{host}:{port}")
    if autoload:
        print(f"   Preloading: {DEFAULT_MODEL_ID}  |  Precision: {DEFAULT_PRECISION}")
    else:
        print("   Idle — waiting for the client to load a model (POST /load).")
    if API_KEY:
        print("   🔒 API key required (X-API-Key header).")
    if ssl_kwargs:
        print("   🔐 TLS enabled (encrypted connection).")
    uvicorn.run(app, host=host, port=port, **ssl_kwargs)


if __name__ == "__main__":
    main()
