#!/usr/bin/env python3
"""
Qwen Image Editor - Client GUI (runs on your personal PC).

This is a thin Tkinter front-end. It does NOT load the model. Instead it sends
the image + parameters to the FastAPI server (see ../server) over HTTP and
displays the returned image. All the heavy GPU work happens on the server.

Configure the server URL via:
  * the "Server" field in the UI, or
  * the QIE_SERVER_URL environment variable (default: http://localhost:8000),
  * optional QIE_API_KEY if the server was started with one.
"""

import base64
import io
import json
import os
import threading
import traceback
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image

try:
    import requests
except ImportError:
    print("❌ 'requests' not installed. Run: pip install -r requirements.txt")
    raise SystemExit(1)

# Optional drag-and-drop support (pip install tkinterdnd2). Degrades gracefully.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False


DEFAULT_SERVER_URL = os.environ.get("QIE_SERVER_URL", "http://193.93.169.217:8000")
DEFAULT_API_KEY = os.environ.get("QIE_API_KEY", "")

# For self-signed TLS, set QIE_SSL_VERIFY=0 so the client accepts the cert
# (traffic is still encrypted; it just skips cert-authority validation).
SSL_VERIFY = os.environ.get("QIE_SSL_VERIFY", "1").lower() not in ("0", "false", "no")
if not SSL_VERIFY:
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass

# Remembers last-used folders across runs.
SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".qwen_image_client.json")
MAX_IMAGES = 3

# Fixed, curated model list. Each entry is (label, model_id, precision, gguf_quant).
# Keeping this closed avoids users typing bad repo ids or unsupported quants.
MODEL_PRESETS = [
    ("Qwen Image Edit 2511 — GGUF Q6_K (recommended)",
     "Qwen/Qwen-Image-Edit-2511", "gguf", "Q6_K"),
    ("Qwen Image Edit 2511 — GGUF Q5_K_M (less VRAM)",
     "Qwen/Qwen-Image-Edit-2511", "gguf", "Q5_K_M"),
    ("Qwen Image Edit 2511 — GGUF Q8_0 (max quality, tight)",
     "Qwen/Qwen-Image-Edit-2511", "gguf", "Q8_0"),
    ("Qwen Image Edit 2511 — full bf16 (best, slow)",
     "Qwen/Qwen-Image-Edit-2511", "max", ""),
    ("Qwen Image Edit 2509 — 4-bit (fast)",
     "Qwen/Qwen-Image-Edit-2509", "4bit", ""),
    ("Qwen Image Edit (original) — 4-bit (fast)",
     "Qwen/Qwen-Image-Edit", "4bit", ""),
]


class ImageEditorClient:
    """Tkinter client for the Qwen Image Edit API."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Qwen Image Editor (Client) v1.1")
        self.root.geometry("1250x720")
        self.root.minsize(980, 600)

        # Up to MAX_IMAGES input slots (image 1 is the main/target).
        self.input_images: list = [None] * MAX_IMAGES
        self.input_paths: list = [""] * MAX_IMAGES
        self.output_image: Optional[Image.Image] = None
        self.is_processing: bool = False
        self._poll_progress: bool = False

        # HTTP session honoring the TLS verify setting.
        self.session = requests.Session()
        self.session.verify = SSL_VERIFY

        # Persisted settings (last-used folders).
        self._settings = self._load_settings()

        self.create_widgets()
        self.setup_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Probe the server once at startup.
        self.check_server()

    # ------------------------------------------------------------------ #
    # Widgets / layout
    # ------------------------------------------------------------------ #
    def create_widgets(self) -> None:
        self.main_frame = ttk.Frame(self.root, padding="10")

        self.title_label = ttk.Label(
            self.main_frame,
            text="🖼️ Qwen Image Editor — Client",
            font=("Arial", 16, "bold"),
        )

        # Server connection bar
        self.server_frame = ttk.LabelFrame(self.main_frame, text="🌐 Server", padding="5")
        ttk.Label(self.server_frame, text="URL:").grid(row=0, column=0, sticky="w")
        self.server_var = tk.StringVar(value=DEFAULT_SERVER_URL)
        self.server_entry = ttk.Entry(self.server_frame, textvariable=self.server_var, width=40)
        ttk.Label(self.server_frame, text="API Key:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.apikey_var = tk.StringVar(value=DEFAULT_API_KEY)
        self.apikey_entry = ttk.Entry(self.server_frame, textvariable=self.apikey_var, width=20, show="*")
        self.connect_btn = ttk.Button(self.server_frame, text="🔌 Check", command=self.check_server)
        self.server_status = ttk.Label(self.server_frame, text="⚪ Not checked", font=("Arial", 9))

        # Model selection bar — a fixed, curated list (read-only).
        self.model_frame = ttk.LabelFrame(self.main_frame, text="🧠 Model", padding="5")
        ttk.Label(self.model_frame, text="Model:").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value="")
        self.model_combo = ttk.Combobox(
            self.model_frame, textvariable=self.model_var, width=52, state="readonly"
        )
        self.load_model_btn = ttk.Button(self.model_frame, text="⬇️ Load", command=self.load_model)
        self.downloads_btn = ttk.Button(self.model_frame, text="🗂 Downloads", command=self.open_downloads)
        self.model_status = ttk.Label(self.model_frame, text="", font=("Arial", 9))
        self._cached_ids: set = set()
        self._refresh_model_choices()  # populate from the fixed list (works offline)

        # LoRA row (optional): a Hugging Face repo id ("repo::file.safetensors")
        # or a direct URL (e.g. a Civitai download link) + a strength.
        ttk.Label(self.model_frame, text="LoRA (URL/repo, optional):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.lora_var = tk.StringVar(value=self._settings.get("lora", ""))
        self.lora_entry = ttk.Entry(self.model_frame, textvariable=self.lora_var, width=52)
        ttk.Label(self.model_frame, text="Strength:").grid(row=1, column=2, sticky="e", pady=(6, 0))
        self.lora_scale_var = tk.DoubleVar(value=float(self._settings.get("lora_scale", 1.0)))
        self.lora_scale_spin = ttk.Spinbox(
            self.model_frame, from_=0.0, to=2.0, increment=0.05,
            textvariable=self.lora_scale_var, width=6,
        )

        # Input images (up to MAX_IMAGES). Slot 0 is the main/target image.
        self.input_frame = ttk.LabelFrame(
            self.main_frame, text="📥 Input Images (1–3)", padding="5"
        )
        self.input_canvases: list = []
        self.input_slot_labels: list = []
        for i in range(MAX_IMAGES):
            slot = ttk.Frame(self.input_frame)
            slot.pack(pady=3, fill=tk.X)
            role = "main" if i == 0 else "optional"
            cv = tk.Canvas(slot, width=250, height=118, bg="#f0f0f0", highlightthickness=1,
                           highlightbackground="#cccccc")
            cv.grid(row=0, column=0, rowspan=2, padx=(0, 6))
            lbl = ttk.Label(slot, text=f"Image {i + 1} ({role})", font=("Arial", 9))
            lbl.grid(row=0, column=1, sticky="sw")
            btns = ttk.Frame(slot)
            btns.grid(row=1, column=1, sticky="nw")
            ttk.Button(btns, text="📂 Browse", width=9,
                       command=lambda idx=i: self.browse_image(idx)).pack(side=tk.LEFT)
            ttk.Button(btns, text="✖", width=3,
                       command=lambda idx=i: self.clear_image(idx)).pack(side=tk.LEFT, padx=3)
            self.input_canvases.append(cv)
            self.input_slot_labels.append(lbl)
            self._register_drop_target(cv, i)
        dnd_hint = "  (tip: drag & drop images onto a slot)" if _DND_AVAILABLE else ""
        self.input_hint = ttk.Label(
            self.input_frame,
            text=("Refer to them as \"image 1/2/3\" in your prompt." + dnd_hint),
            font=("Arial", 8), foreground="#666666", wraplength=270, justify="left",
        )

        # Controls
        self.controls_frame = ttk.LabelFrame(self.main_frame, text="⚙️ Edit Controls", padding="5")
        self.prompt_text = tk.Text(self.controls_frame, width=40, height=3, wrap=tk.WORD, font=("Arial", 10))
        self.prompt_text.insert("1.0", "make him wear cool gaming headphones.")

        self.advanced_frame = ttk.LabelFrame(self.controls_frame, text="🔧 Advanced Settings", padding="3")
        ttk.Label(self.advanced_frame, text="Inference Steps:").grid(row=0, column=0, sticky="w")
        self.steps_var = tk.IntVar(value=30)
        self.steps_spinbox = ttk.Spinbox(self.advanced_frame, from_=1, to=100, textvariable=self.steps_var, width=10)

        ttk.Label(self.advanced_frame, text="CFG Scale:").grid(row=1, column=0, sticky="w")
        self.cfg_var = tk.DoubleVar(value=4.0)
        self.cfg_spinbox = ttk.Spinbox(
            self.advanced_frame, from_=1.0, to=20.0, increment=0.5, textvariable=self.cfg_var, width=10
        )

        ttk.Label(self.advanced_frame, text="Negative Prompt:").grid(row=2, column=0, sticky="w")
        self.neg_prompt_var = tk.StringVar(value="")
        self.neg_prompt_entry = ttk.Entry(self.advanced_frame, textvariable=self.neg_prompt_var, width=30)

        ttk.Label(self.advanced_frame, text="Seed:").grid(row=3, column=0, sticky="w")
        self.seed_var = tk.IntVar(value=0)
        self.seed_spinbox = ttk.Spinbox(self.advanced_frame, from_=0, to=999999999, textvariable=self.seed_var, width=15)

        self.edit_btn = ttk.Button(self.controls_frame, text="🎨 Edit Image", command=self.edit_image)
        self.progress_bar = ttk.Progressbar(self.controls_frame, mode="indeterminate")
        self.status_label = ttk.Label(self.controls_frame, text="Ready.", font=("Arial", 9))

        # Output
        self.output_frame = ttk.LabelFrame(self.main_frame, text="📤 Output Image", padding="5")
        self.output_canvas = tk.Canvas(self.output_frame, width=300, height=300, bg="#f0f0f0")
        self.output_label = ttk.Label(self.output_frame, text="No output yet")
        self.save_btn = ttk.Button(self.output_frame, text="💾 Save Image", command=self.save_image, state="disabled")

    def setup_layout(self) -> None:
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.title_label.pack(pady=(0, 10))

        self.server_frame.pack(fill=tk.X, pady=(0, 10))
        self.server_entry.grid(row=0, column=1, padx=5)
        self.apikey_entry.grid(row=0, column=3, padx=5)
        self.connect_btn.grid(row=0, column=4, padx=5)
        self.server_status.grid(row=0, column=5, padx=10)

        self.model_frame.pack(fill=tk.X, pady=(0, 10))
        self.model_combo.grid(row=0, column=1, padx=5)
        self.load_model_btn.grid(row=0, column=2, padx=5)
        self.downloads_btn.grid(row=0, column=3, padx=2)
        self.model_status.grid(row=0, column=4, padx=10, sticky="w")
        self.lora_entry.grid(row=1, column=1, padx=5, pady=(6, 0), sticky="w")
        self.lora_scale_spin.grid(row=1, column=3, padx=5, pady=(6, 0), sticky="w")

        content_frame = ttk.Frame(self.main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.input_hint.pack(pady=(4, 0))

        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Label(self.controls_frame, text="✏️ Edit Prompt:").pack(pady=(0, 5))
        self.prompt_text.pack(pady=(0, 10))
        self.advanced_frame.pack(pady=5, fill=tk.X)
        self.steps_spinbox.grid(row=0, column=1, padx=5, pady=2, sticky="w")
        self.cfg_spinbox.grid(row=1, column=1, padx=5, pady=2, sticky="w")
        self.neg_prompt_entry.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        self.seed_spinbox.grid(row=3, column=1, padx=5, pady=2, sticky="w")
        self.edit_btn.pack(pady=15)
        self.progress_bar.pack(pady=5, fill=tk.X)
        self.status_label.pack(pady=2)

        self.output_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.output_canvas.pack(pady=5)
        self.output_label.pack(pady=2)
        self.save_btn.pack(pady=5)

    # ------------------------------------------------------------------ #
    # Server helpers
    # ------------------------------------------------------------------ #
    def base_url(self) -> str:
        return self.server_var.get().strip().rstrip("/")

    def auth_headers(self) -> dict:
        key = self.apikey_var.get().strip()
        return {"X-API-Key": key} if key else {}

    def check_server(self) -> None:
        """Ping /health in a background thread and update the status label."""
        def worker():
            url = f"{self.base_url()}/health"
            try:
                resp = self.session.get(url, headers=self.auth_headers(), timeout=10)
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status")
                model = data.get("model_id", "?")
                if status == "ready":
                    text = f"🟢 Ready ({data.get('device')}, {data.get('precision')}) · {model}"
                elif status == "loading":
                    text = f"🟡 Loading… ({data.get('message', '')})"
                elif status == "error":
                    text = f"🔴 Error: {data.get('error')}"
                else:
                    text = f"⚪ {status}"
                self.root.after(0, lambda: self.server_status.config(text=text))
            except Exception as exc:  # noqa: BLE001
                # Bind the message now — `exc` is cleared when the except block
                # ends, but this lambda runs later on the Tk main thread.
                msg = str(exc)
                self.root.after(
                    0, lambda: self.server_status.config(text=f"🔴 Unreachable: {msg}")
                )

        threading.Thread(target=worker, daemon=True).start()
        self._fetch_models()

    # ------------------------------------------------------------------ #
    # Model selection
    # ------------------------------------------------------------------ #
    def _current_preset(self):
        """Return the MODEL_PRESETS tuple for the current dropdown selection."""
        idx = self.model_combo.current()
        if 0 <= idx < len(MODEL_PRESETS):
            return MODEL_PRESETS[idx]
        return None

    def _refresh_model_choices(self) -> None:
        """Rebuild the dropdown labels from the fixed list, marking cached ones.

        ✅ = already downloaded on the server, ⬇ = will download on first load.
        Works offline (before the server is reachable) using the fixed list.
        """
        idx = self.model_combo.current()
        labels = []
        for label, model_id, _precision, _quant in MODEL_PRESETS:
            mark = "✅ " if model_id in self._cached_ids else "⬇ "
            labels.append(mark + label)
        self.model_combo["values"] = labels
        # Preserve the current selection, else default to the first (recommended).
        self.model_combo.current(idx if idx >= 0 else 0)

    def _fetch_models(self) -> None:
        """Fetch the server's cache to mark which fixed presets are downloaded."""
        def worker():
            try:
                resp = self.session.get(
                    f"{self.base_url()}/cache", headers=self.auth_headers(), timeout=10
                )
                resp.raise_for_status()
                cached = {m["repo_id"] for m in resp.json().get("models", [])}
            except Exception:
                return  # offline — keep the plain fixed list
            self.root.after(0, lambda: self._set_cached(cached))

        threading.Thread(target=worker, daemon=True).start()

    def _set_cached(self, cached: set) -> None:
        self._cached_ids = cached
        self._refresh_model_choices()

    def open_downloads(self) -> None:
        """Popup listing downloaded models with sizes and a delete option."""
        win = tk.Toplevel(self.root)
        win.title("🗂 Downloaded Models")
        win.geometry("560x360")
        win.transient(self.root)

        ttk.Label(win, text="Models cached on the server", font=("Arial", 11, "bold")).pack(pady=8)
        listbox = tk.Listbox(win, width=80, height=12)
        listbox.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        status = ttk.Label(win, text="Loading…", font=("Arial", 9))
        status.pack(pady=2)

        rows: list = []  # parallel to listbox: list of (repo_id, size_str)

        def set_status(text: str) -> None:
            # The window may have been closed before a background call returns.
            if win.winfo_exists():
                status.config(text=text)

        def refresh():
            set_status("Loading…")
            def worker():
                try:
                    resp = self.session.get(
                        f"{self.base_url()}/cache", headers=self.auth_headers(), timeout=15
                    )
                    resp.raise_for_status()
                    models = resp.json().get("models", [])
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc)
                    self.root.after(0, lambda: set_status(f"🔴 {msg}"))
                    return

                def apply():
                    if not win.winfo_exists():
                        return
                    rows.clear()
                    listbox.delete(0, tk.END)
                    total = 0
                    for m in models:
                        rows.append((m["repo_id"], m["size_str"]))
                        listbox.insert(tk.END, f"{m['repo_id']}   —   {m['size_str']}")
                        total += m.get("size", 0)
                    gb = total / (1024 ** 3)
                    set_status(f"{len(models)} model(s), {gb:.1f} GB total")
                self.root.after(0, apply)

            threading.Thread(target=worker, daemon=True).start()

        def delete_selected():
            sel = listbox.curselection()
            if not sel:
                return
            repo_id = rows[sel[0]][0]
            if not messagebox.askyesno("Delete", f"Delete '{repo_id}' from the server cache?"):
                return
            set_status(f"Deleting {repo_id}…")
            def worker():
                try:
                    resp = self.session.post(
                        f"{self.base_url()}/cache/delete", json={"repo_id": repo_id},
                        headers=self.auth_headers(), timeout=60,
                    )
                    resp.raise_for_status()
                    freed = resp.json().get("freed_str", "")
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc)
                    self.root.after(0, lambda: set_status(f"🔴 {msg}"))
                    return
                self.root.after(0, lambda: set_status(f"✅ Freed {freed}"))
                self.root.after(300, lambda: refresh() if win.winfo_exists() else None)
                self.root.after(300, self._fetch_models)  # refresh dropdown marks
            threading.Thread(target=worker, daemon=True).start()

        btns = ttk.Frame(win)
        btns.pack(pady=6)
        ttk.Button(btns, text="🔄 Refresh", command=refresh).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text="🗑 Delete selected", command=delete_selected).pack(side=tk.LEFT, padx=5)
        refresh()

    def load_model(self) -> None:
        """POST /load for the selected preset, then poll /health until ready."""
        preset = self._current_preset()
        if preset is None:
            messagebox.showwarning("No Model", "Pick a model from the list first.")
            return
        _label, model_id, precision, gguf_quant = preset

        lora = self.lora_var.get().strip()
        payload = {
            "model_id": model_id,
            "precision": precision,
            "gguf_quant": gguf_quant or None,
            "lora_source": lora or None,
            "lora_scale": self.lora_scale_var.get() if lora else None,
        }
        # Remember the LoRA choice for next launch.
        self._settings["lora"] = lora
        self._settings["lora_scale"] = self.lora_scale_var.get()
        self._save_settings()

        self.load_model_btn.config(state="disabled")
        self.model_status.config(text="⏳ Requesting load…")

        def worker():
            try:
                resp = self.session.post(
                    f"{self.base_url()}/load", json=payload,
                    headers=self.auth_headers(), timeout=30,
                )
                resp.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                self.root.after(0, lambda: self._load_done(f"🔴 Load failed: {msg}"))
                return

            # Poll /health until ready/error (model download can take a while).
            import time as _t
            deadline = _t.time() + 3600
            while _t.time() < deadline:
                try:
                    h = self.session.get(
                        f"{self.base_url()}/health", headers=self.auth_headers(), timeout=10
                    ).json()
                except Exception:
                    _t.sleep(3)
                    continue
                st = h.get("status")
                if st == "ready":
                    self.root.after(0, lambda: self._load_done(
                        f"🟢 Loaded: {h.get('model_id')} ({h.get('precision')})"))
                    self.root.after(0, self.check_server)
                    return
                if st == "error":
                    err = h.get("error", "unknown")
                    self.root.after(0, lambda: self._load_done(f"🔴 Error: {err}"))
                    return
                self.root.after(0, lambda m=h.get("message", ""): self.model_status.config(
                    text=f"🟡 Loading… {m}"))
                _t.sleep(4)
            self.root.after(0, lambda: self._load_done("🔴 Timed out waiting for load."))

        threading.Thread(target=worker, daemon=True).start()

    def _load_done(self, text: str) -> None:
        self.model_status.config(text=text)
        self.load_model_btn.config(state="normal")

    # ------------------------------------------------------------------ #
    # Settings persistence (remembered folders)
    # ------------------------------------------------------------------ #
    def _load_settings(self) -> dict:
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _save_settings(self) -> None:
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
                json.dump(self._settings, fh)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Drag & drop (optional; needs tkinterdnd2)
    # ------------------------------------------------------------------ #
    def _register_drop_target(self, canvas: tk.Canvas, idx: int) -> None:
        if not _DND_AVAILABLE:
            return
        try:
            canvas.drop_target_register(DND_FILES)
            canvas.dnd_bind("<<Drop>>", lambda e, i=idx: self._on_drop(e, i))
        except Exception:
            pass

    def _on_drop(self, event, idx: int) -> None:
        # event.data may be "{C:/path with spaces.png}" or multiple paths.
        data = event.data.strip()
        path = None
        if data.startswith("{"):
            path = data[1:data.index("}")] if "}" in data else data.strip("{}")
        else:
            path = data.split()[0] if data else None
        if path:
            self.load_image(path, idx)

    # ------------------------------------------------------------------ #
    # Image loading / display
    # ------------------------------------------------------------------ #
    def browse_image(self, idx: int = 0) -> None:
        filetypes = (
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp *.gif"),
            ("All files", "*.*"),
        )
        filename = filedialog.askopenfilename(
            title=f"Select image {idx + 1}",
            initialdir=self._settings.get("input_dir", os.getcwd()),
            filetypes=filetypes,
        )
        if filename:
            self.load_image(filename, idx)

    def load_image(self, filepath: str, idx: int = 0) -> None:
        try:
            img = Image.open(filepath).convert("RGB")
            self.input_images[idx] = img
            self.input_paths[idx] = filepath
            self._show_on_canvas(self.input_canvases[idx], img, size=(250, 118))
            w, h = img.size
            role = "main" if idx == 0 else "optional"
            self.input_slot_labels[idx].config(
                text=f"Image {idx + 1} ({role}) · {os.path.basename(filepath)} · {w}×{h}"
            )
            self._settings["input_dir"] = os.path.dirname(filepath)
            self._save_settings()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Image Loading Error", f"Failed to load image:\n{exc}")

    def clear_image(self, idx: int) -> None:
        self.input_images[idx] = None
        self.input_paths[idx] = ""
        self.input_canvases[idx].delete("all")
        role = "main" if idx == 0 else "optional"
        self.input_slot_labels[idx].config(text=f"Image {idx + 1} ({role})")

    def _show_on_canvas(self, canvas: tk.Canvas, image: Image.Image, size=(300, 300)) -> None:
        display = self._resize_for_display(image, size[0], size[1])
        buffer = io.BytesIO()
        display.save(buffer, format="PNG")
        photo = tk.PhotoImage(data=base64.b64encode(buffer.getvalue()))
        canvas.delete("all")
        canvas.create_image(size[0] // 2, size[1] // 2, image=photo)
        canvas.image = photo  # keep a reference

    @staticmethod
    def _resize_for_display(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
        w, h = image.size
        scale = min(max_w / w, max_h / h)
        if scale < 1:
            new_size = (int(w * scale), int(h * scale))
            try:
                return image.resize(new_size, Image.Resampling.LANCZOS)
            except AttributeError:
                return image.resize(new_size, Image.LANCZOS)
        return image

    # ------------------------------------------------------------------ #
    # Edit (calls the API)
    # ------------------------------------------------------------------ #
    def edit_image(self) -> None:
        images = [im for im in self.input_images if im is not None]
        if not images:
            messagebox.showwarning("No Input Image", "Please select at least one input image.")
            return
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("No Prompt", "Please enter an edit prompt.")
            return
        if self.is_processing:
            return

        self.is_processing = True
        self.edit_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.progress_bar.config(mode="indeterminate")
        self.progress_bar.start()
        n = len(images)
        self.status_label.config(text=f"📤 Uploading {n} image{'s' if n > 1 else ''} to server…")

        params = {
            "prompt": prompt,
            "num_inference_steps": self.steps_var.get(),
            "true_cfg_scale": self.cfg_var.get(),
            "negative_prompt": self.neg_prompt_var.get(),
            "seed": self.seed_var.get(),
        }

        # Start polling the server for live inference progress.
        self._poll_progress = True
        threading.Thread(target=self._progress_poller, daemon=True).start()
        threading.Thread(target=self._request_edit, args=(params,), daemon=True).start()

    def _progress_poller(self) -> None:
        """Poll /progress while an edit runs and reflect stage/step in the UI."""
        import time as _t
        while self._poll_progress:
            try:
                data = self.session.get(
                    f"{self.base_url()}/progress", headers=self.auth_headers(), timeout=5
                ).json()
            except Exception:
                _t.sleep(0.6)
                continue
            if not self._poll_progress:
                break
            self.root.after(0, lambda d=data: self._apply_progress(d))
            _t.sleep(0.6)

    def _apply_progress(self, data: dict) -> None:
        if not self.is_processing:
            return
        stage = data.get("stage", "")
        step = data.get("step", 0)
        total = data.get("total", 0)
        if stage == "denoising" and total:
            # Switch to a determinate bar showing step X / Y.
            if self.progress_bar["mode"] != "determinate":
                self.progress_bar.stop()
                self.progress_bar.config(mode="determinate", maximum=total)
            self.progress_bar["value"] = step
            pct = int(step / total * 100)
            self.status_label.config(text=f"🎨 Denoising step {step}/{total} ({pct}%)")
        elif stage == "preparing":
            self.status_label.config(text="🧠 Server preparing (encoding prompt/image)…")

    def _request_edit(self, params: dict) -> None:
        try:
            # Send every filled slot under the repeated "image" field (in order,
            # so the server sees image 1, 2, 3 as referenced in the prompt).
            files = []
            for i, img in enumerate(im for im in self.input_images if im is not None):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                files.append(("image", (f"image{i + 1}.png", buf, "image/png")))

            resp = self.session.post(
                f"{self.base_url()}/edit",
                data=params,
                files=files,
                headers=self.auth_headers(),
                timeout=600,  # inference can take a while
            )

            if resp.status_code == 503:
                raise RuntimeError(
                    "Server model not ready yet. Wait for it to finish loading and try again."
                )
            resp.raise_for_status()

            result = Image.open(io.BytesIO(resp.content)).convert("RGB")
            self.output_image = result
            self.root.after(0, self._edit_success)
        except Exception as exc:  # noqa: BLE001
            print(traceback.format_exc())
            msg = str(exc)
            self.root.after(0, lambda: self._edit_error(msg))

    def _reset_progress_bar(self) -> None:
        self._poll_progress = False
        self.progress_bar.stop()
        self.progress_bar.config(mode="indeterminate")
        self.progress_bar["value"] = 0

    def _edit_success(self) -> None:
        self.is_processing = False
        self._reset_progress_bar()
        self._show_on_canvas(self.output_canvas, self.output_image)
        w, h = self.output_image.size
        self.output_label.config(text=f"✨ Edited Image\n📐 {w}×{h} pixels")
        self.status_label.config(text="✅ Done.")
        self.edit_btn.config(state="normal")
        self.save_btn.config(state="normal")

    def _edit_error(self, msg: str) -> None:
        self.is_processing = False
        self._reset_progress_bar()
        self.status_label.config(text="❌ Failed.")
        self.edit_btn.config(state="normal")
        messagebox.showerror("Edit Failed", f"Could not edit image:\n\n{msg}")

    # ------------------------------------------------------------------ #
    # Save / close
    # ------------------------------------------------------------------ #
    def save_image(self) -> None:
        if self.output_image is None:
            messagebox.showwarning("No Output", "No edited image to save.")
            return
        main_path = self.input_paths[0]
        base = (
            os.path.splitext(os.path.basename(main_path))[0]
            if main_path
            else "edited_image"
        )
        filename = filedialog.asksaveasfilename(
            title="Save edited image",
            initialdir=self._settings.get("output_dir", self._settings.get("input_dir", os.getcwd())),
            initialfile=f"{base}_edited.png",
            defaultextension=".png",
            filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg *.jpeg"), ("All files", "*.*")),
        )
        if filename:
            try:
                self.output_image.save(filename)
                self._settings["output_dir"] = os.path.dirname(filename)
                self._save_settings()
                messagebox.showinfo("Saved", f"✅ Saved to:\n{os.path.abspath(filename)}")
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("Save Error", f"Failed to save image:\n{exc}")

    def on_closing(self) -> None:
        if self.is_processing:
            if messagebox.askokcancel("Quit", "An edit is in progress. Quit anyway?"):
                self.root.destroy()
        else:
            self.root.destroy()


def main() -> None:
    # Use the DnD-capable root if tkinterdnd2 is installed, else plain Tk.
    root = TkinterDnD.Tk() if _DND_AVAILABLE else tk.Tk()
    root.resizable(True, True)
    ImageEditorClient(root)
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (w // 2)
    y = (root.winfo_screenheight() // 2) - (h // 2)
    root.geometry(f"{w}x{h}+{x}+{y}")
    root.mainloop()


if __name__ == "__main__":
    main()
