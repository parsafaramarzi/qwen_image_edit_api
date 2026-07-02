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
    # Load in the background so the server can answer /health immediately and
    # report loading progress instead of blocking the boot.
    manager.load_in_background()


def _check_api_key(provided: str | None) -> None:
    if API_KEY and provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@app.get("/health")
@app.get("/status")
def health() -> JSONResponse:
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
    )
    return JSONResponse(manager.state.as_dict())


@app.post("/edit")
async def edit(
    image: UploadFile = File(..., description="Input image to edit."),
    prompt: str = Form(..., description="Edit instruction."),
    num_inference_steps: int = Form(40),
    true_cfg_scale: float = Form(4.0),
    negative_prompt: str = Form(""),
    seed: int = Form(0),
    guidance_scale: float = Form(1.0),
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

    try:
        raw = await image.read()
        pil_image = Image.open(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    try:
        result = manager.edit(
            pil_image,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            seed=seed,
            guidance_scale=guidance_scale,
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
    print(f"🚀 Starting Qwen Image Edit API on http://{host}:{port}")
    print(f"   Model: {DEFAULT_MODEL_ID}  |  Precision: {DEFAULT_PRECISION}")
    if API_KEY:
        print("   🔒 API key required (X-API-Key header).")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
