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


DEFAULT_SERVER_URL = os.environ.get("QIE_SERVER_URL", "http://193.93.169.217:8000")
DEFAULT_API_KEY = os.environ.get("QIE_API_KEY", "")


class ImageEditorClient:
    """Tkinter client for the Qwen Image Edit API."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Qwen Image Editor (Client) v1.0")
        self.root.geometry("1200x680")
        self.root.minsize(900, 560)

        self.input_image: Optional[Image.Image] = None
        self.output_image: Optional[Image.Image] = None
        self.input_image_path: str = ""
        self.is_processing: bool = False

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

        # Model selection bar
        self.model_frame = ttk.LabelFrame(self.main_frame, text="🧠 Model", padding="5")
        ttk.Label(self.model_frame, text="Model:").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value="")
        # Editable combobox: pick a preset or type any HF repo id.
        self.model_combo = ttk.Combobox(self.model_frame, textvariable=self.model_var, width=42)
        ttk.Label(self.model_frame, text="Precision:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.precision_var = tk.StringVar(value="gguf")
        self.precision_combo = ttk.Combobox(
            self.model_frame, textvariable=self.precision_var, width=8, state="readonly",
            values=["gguf", "4bit", "8bit", "bf16", "max"],
        )
        ttk.Label(self.model_frame, text="GGUF quant:").grid(row=0, column=4, sticky="w", padx=(10, 0))
        self.gguf_quant_var = tk.StringVar(value="Q6_K")
        self.gguf_quant_entry = ttk.Entry(self.model_frame, textvariable=self.gguf_quant_var, width=8)
        self.load_model_btn = ttk.Button(self.model_frame, text="⬇️ Load Model", command=self.load_model)
        self.model_status = ttk.Label(self.model_frame, text="", font=("Arial", 9))
        # Maps the combobox label → (model_id, precision) from /models.
        self._model_presets: dict = {}
        # When a preset is chosen, sync the precision box to its suggestion.
        self.model_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        # Input
        self.input_frame = ttk.LabelFrame(self.main_frame, text="📥 Input Image", padding="5")
        self.input_canvas = tk.Canvas(self.input_frame, width=300, height=300, bg="#f0f0f0")
        self.input_label = ttk.Label(self.input_frame, text="No image selected")
        self.browse_btn = ttk.Button(self.input_frame, text="📂 Browse Image", command=self.browse_image)

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
        self.precision_combo.grid(row=0, column=3, padx=5)
        self.gguf_quant_entry.grid(row=0, column=5, padx=5)
        self.load_model_btn.grid(row=0, column=6, padx=5)
        self.model_status.grid(row=0, column=7, padx=10)

        content_frame = ttk.Frame(self.main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.input_canvas.pack(pady=5)
        self.input_label.pack(pady=2)
        self.browse_btn.pack(pady=5)

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
                resp = requests.get(url, headers=self.auth_headers(), timeout=10)
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
    def _fetch_models(self) -> None:
        """Populate the model dropdown from the server's /models list."""
        def worker():
            try:
                resp = requests.get(
                    f"{self.base_url()}/models", headers=self.auth_headers(), timeout=10
                )
                resp.raise_for_status()
                items = resp.json().get("models", [])
            except Exception:
                return  # leave the combobox as free-text only

            presets = {}
            labels = []
            for it in items:
                label = it.get("label") or it.get("model_id", "")
                presets[label] = (it.get("model_id", ""), it.get("precision", "gguf"))
                labels.append(label)

            def apply():
                self._model_presets = presets
                self.model_combo["values"] = labels
                # Prefill with the first suggestion if nothing chosen yet.
                if labels and not self.model_var.get():
                    self.model_combo.set(labels[0])
                    self._on_preset_selected()
            self.root.after(0, apply)

        threading.Thread(target=worker, daemon=True).start()

    def _on_preset_selected(self, _event=None) -> None:
        """When a preset label is picked, sync precision to its suggestion."""
        preset = self._model_presets.get(self.model_var.get())
        if preset:
            self.precision_var.set(preset[1])

    def _selected_model_id(self) -> str:
        """Resolve the field to a model id (a preset label maps to its repo id)."""
        value = self.model_var.get().strip()
        preset = self._model_presets.get(value)
        return preset[0] if preset else value

    def load_model(self) -> None:
        """POST /load then poll /health until the new model is ready."""
        model_id = self._selected_model_id()
        if not model_id:
            messagebox.showwarning("No Model", "Pick a preset or type a model id first.")
            return

        payload = {
            "model_id": model_id,
            "precision": self.precision_var.get().strip(),
            "gguf_quant": self.gguf_quant_var.get().strip() or None,
        }
        self.load_model_btn.config(state="disabled")
        self.model_status.config(text="⏳ Requesting load…")

        def worker():
            try:
                resp = requests.post(
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
                    h = requests.get(
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
    # Image loading / display
    # ------------------------------------------------------------------ #
    def browse_image(self) -> None:
        filetypes = (
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp *.gif"),
            ("All files", "*.*"),
        )
        filename = filedialog.askopenfilename(
            title="Select an image to edit", initialdir=os.getcwd(), filetypes=filetypes
        )
        if filename:
            self.load_image(filename)

    def load_image(self, filepath: str) -> None:
        try:
            self.input_image = Image.open(filepath).convert("RGB")
            self.input_image_path = filepath
            self._show_on_canvas(self.input_canvas, self.input_image)
            w, h = self.input_image.size
            self.input_label.config(text=f"📄 {os.path.basename(filepath)}\n📐 {w}×{h} pixels")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Image Loading Error", f"Failed to load image:\n{exc}")
            self.input_image = None
            self.input_image_path = ""

    def _show_on_canvas(self, canvas: tk.Canvas, image: Image.Image) -> None:
        display = self._resize_for_display(image, 300, 300)
        buffer = io.BytesIO()
        display.save(buffer, format="PNG")
        photo = tk.PhotoImage(data=base64.b64encode(buffer.getvalue()))
        canvas.delete("all")
        canvas.create_image(150, 150, image=photo)
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
        if self.input_image is None:
            messagebox.showwarning("No Input Image", "Please select an input image first.")
            return
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("No Prompt", "Please enter an edit prompt.")
            return
        if self.is_processing:
            return

        self.is_processing = True
        self.edit_btn.config(state="disabled")
        self.browse_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.progress_bar.start()
        self.status_label.config(text="🎨 Sending to server…")

        params = {
            "prompt": prompt,
            "num_inference_steps": self.steps_var.get(),
            "true_cfg_scale": self.cfg_var.get(),
            "negative_prompt": self.neg_prompt_var.get(),
            "seed": self.seed_var.get(),
        }

        threading.Thread(target=self._request_edit, args=(params,), daemon=True).start()

    def _request_edit(self, params: dict) -> None:
        try:
            buffer = io.BytesIO()
            self.input_image.save(buffer, format="PNG")
            buffer.seek(0)
            files = {"image": ("input.png", buffer, "image/png")}

            resp = requests.post(
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

    def _edit_success(self) -> None:
        self.is_processing = False
        self.progress_bar.stop()
        self._show_on_canvas(self.output_canvas, self.output_image)
        w, h = self.output_image.size
        self.output_label.config(text=f"✨ Edited Image\n📐 {w}×{h} pixels")
        self.status_label.config(text="✅ Done.")
        self.edit_btn.config(state="normal")
        self.browse_btn.config(state="normal")
        self.save_btn.config(state="normal")

    def _edit_error(self, msg: str) -> None:
        self.is_processing = False
        self.progress_bar.stop()
        self.status_label.config(text="❌ Failed.")
        self.edit_btn.config(state="normal")
        self.browse_btn.config(state="normal")
        messagebox.showerror("Edit Failed", f"Could not edit image:\n\n{msg}")

    # ------------------------------------------------------------------ #
    # Save / close
    # ------------------------------------------------------------------ #
    def save_image(self) -> None:
        if self.output_image is None:
            messagebox.showwarning("No Output", "No edited image to save.")
            return
        base = (
            os.path.splitext(os.path.basename(self.input_image_path))[0]
            if self.input_image_path
            else "edited_image"
        )
        filename = filedialog.asksaveasfilename(
            title="Save edited image",
            initialfile=f"{base}_edited.png",
            defaultextension=".png",
            filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg *.jpeg"), ("All files", "*.*")),
        )
        if filename:
            try:
                self.output_image.save(filename)
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
    root = tk.Tk()
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
