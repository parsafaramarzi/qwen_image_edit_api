#!/usr/bin/env python3
"""
Qwen Image Studio — web chat client (Streamlit).

A ChatGPT/Gemini-style front-end for the Qwen Image Edit/Generate API server.
It does NOT run any model — it talks to the FastAPI server over HTTP.

Features
--------
* Chat interface with a prompt box + drag-and-drop image upload (native).
* Model + LoRA controls, precision, and generation settings in the sidebar.
* Multiple chats, saved/cached locally, with a "New chat" button.
* Click an output image → "Use as input" to feed it into the next prompt,
  with no manual saving.
* Works with BOTH editing models (image in → image out) and generation models
  (prompt only → image). Sending an image to a generation model is harmless.

Run
---
    streamlit run chat_app.py

Config via env vars: QIE_SERVER_URL, QIE_API_KEY, QIE_SSL_VERIFY.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
import uuid
from datetime import datetime

import requests
import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------- #
# Config + local storage
# --------------------------------------------------------------------------- #
DEFAULT_SERVER_URL = os.environ.get("QIE_SERVER_URL", "https://193.93.169.217:8000")
DEFAULT_API_KEY = os.environ.get("QIE_API_KEY", "")
SSL_VERIFY = os.environ.get("QIE_SSL_VERIFY", "1").lower() not in ("0", "false", "no")
if not SSL_VERIFY:
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass

DATA_DIR = os.path.join(os.path.expanduser("~"), ".qwen_chat_ui")
CHATS_DIR = os.path.join(DATA_DIR, "chats")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(CHATS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

MODEL_PRESETS = [
    ("Qwen Image Edit 2511 — GGUF Q6_K (recommended)", "Qwen/Qwen-Image-Edit-2511", "gguf", "Q6_K"),
    ("Qwen Image Edit 2511 — GGUF Q5_K_M (less VRAM)", "Qwen/Qwen-Image-Edit-2511", "gguf", "Q5_K_M"),
    ("Qwen Image Edit 2511 — full bf16 (best, slow)", "Qwen/Qwen-Image-Edit-2511", "max", ""),
    ("Qwen Image Edit 2509 — 4-bit (fast)", "Qwen/Qwen-Image-Edit-2509", "4bit", ""),
    ("Qwen Image Edit (original) — 4-bit", "Qwen/Qwen-Image-Edit", "4bit", ""),
    ("➕ Custom model…", "", "4bit", ""),
]


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def _url(path: str) -> str:
    return f"{st.session_state.server_url.rstrip('/')}{path}"


def _headers() -> dict:
    key = st.session_state.api_key.strip()
    return {"X-API-Key": key} if key else {}


def api_get(path: str, timeout: int = 15):
    return requests.get(_url(path), headers=_headers(), verify=SSL_VERIFY, timeout=timeout)


def api_post(path: str, timeout: int = 30, **kw):
    return requests.post(_url(path), headers=_headers(), verify=SSL_VERIFY, timeout=timeout, **kw)


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Chat persistence
# --------------------------------------------------------------------------- #
def _chat_file(cid: str) -> str:
    return os.path.join(CHATS_DIR, f"{cid}.json")


def save_chat(chat: dict) -> None:
    with open(_chat_file(chat["id"]), "w", encoding="utf-8") as fh:
        json.dump(chat, fh)


def load_all_chats() -> list:
    chats = []
    for fn in os.listdir(CHATS_DIR):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(CHATS_DIR, fn), "r", encoding="utf-8") as fh:
                    chats.append(json.load(fh))
            except Exception:
                pass
    return sorted(chats, key=lambda c: c.get("created", ""), reverse=True)


def new_chat() -> dict:
    cid = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    chat = {"id": cid, "title": "New chat", "created": datetime.now().isoformat(), "messages": []}
    save_chat(chat)
    return chat


def delete_chat(cid: str) -> None:
    try:
        os.remove(_chat_file(cid))
    except Exception:
        pass


def save_image(cid: str, img: Image.Image) -> str:
    d = os.path.join(IMAGES_DIR, cid)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, uuid.uuid4().hex + ".png")
    img.convert("RGB").save(path)
    return path


# --------------------------------------------------------------------------- #
# Server actions
# --------------------------------------------------------------------------- #
def poll_until(pred_done, status_box, get_state, deadline_s: int = 3600):
    """Poll get_state() until pred_done(state) or timeout, updating status_box."""
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            state = get_state()
        except Exception:
            time.sleep(2)
            continue
        done, text = pred_done(state)
        status_box.info(text)
        if done:
            return state
        time.sleep(2)
    return None


def action_load_model(model_id, precision, gguf_quant, lora, lora_scale, status_box):
    payload = {
        "model_id": model_id,
        "precision": precision,
        "gguf_quant": gguf_quant or None,
        "lora_source": lora or None,
        "lora_scale": lora_scale if lora else None,
    }
    try:
        api_post("/load", json=payload, timeout=30).raise_for_status()
    except Exception as exc:  # noqa: BLE001
        status_box.error(f"Load request failed: {exc}")
        return

    def pred(state):
        s = state.get("status")
        if s == "ready":
            return True, f"🟢 Ready: {state.get('model_id')} ({state.get('precision')})"
        if s == "error":
            return True, f"🔴 Error: {state.get('error')}"
        return False, f"🟡 Loading… {state.get('message', '')}"

    poll_until(pred, status_box, lambda: api_get("/health").json())


def action_generate(images, prompt, params, prog_box):
    """Run one generation/edit with live progress. Returns (PIL image, error)."""
    out = {"img": None, "err": None}

    def worker():
        try:
            files = [("image", (f"img{i}.png", _png_bytes(im), "image/png"))
                     for i, im in enumerate(images)]
            resp = api_post("/edit", data=params, files=files or None, timeout=1800)
            if resp.status_code == 503:
                out["err"] = "Model not ready — load a model in the sidebar first."
                return
            resp.raise_for_status()
            out["img"] = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            out["err"] = str(exc)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    bar = prog_box.progress(0.0, text="Sending to server…")
    while t.is_alive():
        try:
            pr = api_get("/progress", timeout=5).json()
            total = pr.get("total") or 0
            if pr.get("stage") == "denoising" and total:
                frac = min(pr.get("step", 0) / total, 1.0)
                bar.progress(frac, text=f"Denoising {pr.get('step')}/{total}")
            else:
                bar.progress(0.05, text=f"{pr.get('stage', 'working')}…")
        except Exception:
            pass
        time.sleep(0.5)
    t.join()
    prog_box.empty()
    return out["img"], out["err"]


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state():
    # Explicit "if key not in state" — st.session_state.setdefault() is not
    # reliable in the live runtime, so set each key directly.
    ss = st.session_state
    defaults = {
        "server_url": DEFAULT_SERVER_URL,
        "api_key": DEFAULT_API_KEY,
        "pending": [],          # image file paths queued as input for next prompt
        "uploader_key": 0,      # bumped to reset the file_uploader
        "custom_model": "",
        "lora": "",
        "lora_scale": 1.0,
        "settings": {
            "steps": 40, "cfg": 4.0, "seed": 0, "negative": "",
            "width": 0, "height": 0,
        },
    }
    for key, value in defaults.items():
        if key not in ss:
            ss[key] = value
    if "current_chat_id" not in ss:
        chats = load_all_chats() or [new_chat()]
        ss["current_chat_id"] = chats[0]["id"]


def current_chat() -> dict:
    for c in load_all_chats():
        if c["id"] == st.session_state.current_chat_id:
            return c
    # Fell through (deleted) — make a fresh one.
    c = new_chat()
    st.session_state.current_chat_id = c["id"]
    return c


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Qwen Image Studio", page_icon="🎨", layout="wide")
init_state()


def sidebar():
    ss = st.session_state
    with st.sidebar:
        st.header("🎨 Qwen Image Studio")

        # --- Chats ---
        if st.button("➕ New chat", use_container_width=True):
            c = new_chat()
            ss.current_chat_id = c["id"]
            ss.pending = []
            st.rerun()

        st.caption("Chats")
        for c in load_all_chats():
            cols = st.columns([0.8, 0.2])
            label = ("• " if c["id"] == ss.current_chat_id else "") + (c.get("title") or "Chat")
            if cols[0].button(label[:34], key=f"chat_{c['id']}", use_container_width=True):
                ss.current_chat_id = c["id"]
                ss.pending = []
                st.rerun()
            if cols[1].button("🗑", key=f"del_{c['id']}"):
                delete_chat(c["id"])
                if ss.current_chat_id == c["id"]:
                    remaining = load_all_chats()
                    ss.current_chat_id = remaining[0]["id"] if remaining else new_chat()["id"]
                st.rerun()

        st.divider()

        # --- Server ---
        with st.expander("🌐 Server", expanded=False):
            ss.server_url = st.text_input("URL", ss.server_url)
            ss.api_key = st.text_input("API key", ss.api_key, type="password")
            if st.button("🔌 Check", use_container_width=True):
                try:
                    h = api_get("/health").json()
                    task = h.get("task", {})
                    st.success(
                        f"{h.get('status')} · {h.get('model_id') or 'no model'} "
                        f"· {task.get('task', '?')}"
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Unreachable: {exc}")

        # --- Model ---
        st.subheader("🧠 Model")
        labels = [p[0] for p in MODEL_PRESETS]
        choice = st.selectbox("Model", labels, index=0)
        preset = MODEL_PRESETS[labels.index(choice)]
        is_custom = preset[1] == ""
        if is_custom:
            model_id = st.text_input("Custom repo id", ss.get("custom_model", ""))
            ss.custom_model = model_id
            precision = st.selectbox("Precision", ["gguf", "4bit", "8bit", "bf16", "max"], index=1)
            gguf_quant = "Q6_K"
        else:
            model_id, precision, gguf_quant = preset[1], preset[2], preset[3]

        lora = st.text_input("LoRA (URL/repo, optional)", ss.get("lora", ""))
        ss.lora = lora
        lora_scale = st.slider("LoRA strength", 0.0, 2.0, float(ss.get("lora_scale", 1.0)), 0.05)
        ss.lora_scale = lora_scale

        status_box = st.empty()
        c1, c2 = st.columns(2)
        if c1.button("⬇️ Load", use_container_width=True):
            action_load_model(model_id, precision, gguf_quant, lora, lora_scale, status_box)
        if c2.button("⏏ Unload", use_container_width=True):
            try:
                api_post("/model/unload").raise_for_status()
                status_box.info("Model unloaded.")
            except Exception as exc:  # noqa: BLE001
                status_box.error(str(exc))

        with st.expander("🎨 LoRA on loaded model"):
            lc1, lc2 = st.columns(2)
            if lc1.button("Load LoRA", use_container_width=True) and lora:
                try:
                    api_post("/lora/load", json={"lora_source": lora, "lora_scale": lora_scale}).raise_for_status()
                    st.info("LoRA requested — check status.")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
            if lc2.button("Unload LoRA", use_container_width=True):
                try:
                    api_post("/lora/unload").raise_for_status()
                    st.info("LoRA removed.")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        # --- Settings ---
        st.subheader("⚙️ Settings")
        s = ss.settings
        s["steps"] = st.slider("Inference steps", 1, 100, s["steps"])
        s["cfg"] = st.slider("CFG scale", 1.0, 20.0, s["cfg"], 0.5)
        s["negative"] = st.text_input("Negative prompt", s["negative"])
        s["seed"] = st.number_input("Seed", 0, 999999999, s["seed"])
        with st.expander("Generation size (text-to-image only)"):
            s["width"] = st.number_input("Width (0 = default)", 0, 2048, s["width"], step=64)
            s["height"] = st.number_input("Height (0 = default)", 0, 2048, s["height"], step=64)


def render_messages(chat: dict):
    for msg in chat["messages"]:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            if msg.get("text"):
                st.write(msg["text"])
            imgs = [p for p in msg.get("images", []) if os.path.exists(p)]
            if imgs:
                cols = st.columns(min(len(imgs), 3))
                for i, p in enumerate(imgs):
                    with cols[i % len(cols)]:
                        st.image(p, use_container_width=True)
                        if msg["role"] == "assistant":
                            if st.button("↩ Use as input", key=f"use_{p}"):
                                if p not in st.session_state.pending:
                                    st.session_state.pending.append(p)
                                st.rerun()
                            with open(p, "rb") as fh:
                                st.download_button("💾 Download", fh.read(),
                                                   file_name=os.path.basename(p), key=f"dl_{p}")
            if msg.get("error"):
                st.error(msg["error"])


def input_area(chat: dict):
    ss = st.session_state

    # Pending input previews (queued images for the next prompt).
    if ss.pending:
        st.caption("Attached inputs for the next prompt:")
        cols = st.columns(min(len(ss.pending), 6))
        for i, p in enumerate(list(ss.pending)):
            with cols[i % len(cols)]:
                if os.path.exists(p):
                    st.image(p, use_container_width=True)
                if st.button("✖", key=f"rm_{i}_{p}"):
                    ss.pending.remove(p)
                    st.rerun()

    uploaded = st.file_uploader(
        "Drag & drop or browse images (optional — leave empty for text-to-image)",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=True,
        key=f"uploader_{ss.uploader_key}",
    )
    prompt = st.chat_input("Describe the edit, or a prompt to generate…")

    if prompt is None:
        return

    # Assemble input images: queued "use as input" + freshly uploaded.
    images = []
    input_paths = []
    for p in ss.pending:
        if os.path.exists(p):
            images.append(Image.open(p).convert("RGB"))
            input_paths.append(p)
    for uf in (uploaded or []):
        try:
            img = Image.open(uf).convert("RGB")
            images.append(img)
            input_paths.append(save_image(chat["id"], img))
        except Exception:
            pass

    # Record the user message.
    chat["messages"].append({"role": "user", "text": prompt, "images": input_paths})
    if chat.get("title") in (None, "", "New chat"):
        chat["title"] = prompt[:40]
    save_chat(chat)

    # Run it.
    params = {
        "prompt": prompt,
        "num_inference_steps": ss.settings["steps"],
        "true_cfg_scale": ss.settings["cfg"],
        "negative_prompt": ss.settings["negative"],
        "seed": ss.settings["seed"],
        "width": ss.settings["width"],
        "height": ss.settings["height"],
    }
    prog_box = st.empty()
    result, err = action_generate(images, prompt, params, prog_box)

    if err:
        chat["messages"].append({"role": "assistant", "text": "", "images": [], "error": err})
    else:
        out_path = save_image(chat["id"], result)
        chat["messages"].append({"role": "assistant", "text": "", "images": [out_path]})
    save_chat(chat)

    # Clear pending inputs + reset the uploader for the next turn.
    ss.pending = []
    ss.uploader_key += 1
    st.rerun()


def main():
    sidebar()
    chat = current_chat()
    st.title(chat.get("title") or "New chat")
    render_messages(chat)
    input_area(chat)


main()
