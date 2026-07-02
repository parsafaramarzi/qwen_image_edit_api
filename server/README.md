# 🖼️ Qwen Image Editor — Server

FastAPI inference server for the **Qwen Image Edit** model. Runs on your GPU
box (the RTX 3090 machine) and exposes an HTTP API that the
[client](../client) drives from another PC.

## Why the old version crashed (and how this fixes it)

The previous code did:

```python
pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")  # loads FP32!
pipeline.to(torch.bfloat16)   # convert AFTER — too late
pipeline.to("cuda")           # whole model onto 24 GB at once
```

Two problems on a 32 GB-RAM / 24 GB-VRAM box:

1. Without `torch_dtype`, the shards load in **float32** first (~110 GB for this
   ~20B transformer + 7B text encoder). That fp32 spike pinned RAM to 99% and
   killed the process **before** the bf16 conversion could help.
2. `.to("cuda")` tried to place the entire ~55 GB (bf16) model on the 3090.

This server fixes both (see [`model_manager.py`](model_manager.py)):

- Loads weights **directly** in `torch.bfloat16` with `low_cpu_mem_usage=True`
  — no fp32 spike.
- Loads the transformer + text encoder in **4-bit (nf4)** by default via
  bitsandbytes → ~14–16 GB resident, fits the 3090.
- Uses `enable_model_cpu_offload()` so components stream to the GPU only while
  in use, keeping peak VRAM low.

### Precision modes (`QIE_PRECISION`)

| Mode   | Approx. footprint | Fits 3090 (24 GB) + 32 GB RAM? | Notes                     |
| ------ | ----------------- | ------------------------------ | ------------------------- |
| `4bit` | ~14–16 GB         | ✅ (default)                    | Best fit, minor quality Δ |
| `8bit` | ~24–28 GB         | ⚠️ tight                       | Needs headroom            |
| `bf16` | ~55 GB            | ❌                              | Large-VRAM machines only  |

## Setup — Ubuntu (recommended for the GPU box)

Tested on **Ubuntu 22.04 / 24.04 LTS** with an NVIDIA RTX 3090.

```bash
# ── 1. NVIDIA driver (skip if `nvidia-smi` already works) ──────────────
sudo apt update
sudo ubuntu-drivers autoinstall      # or: sudo apt install nvidia-driver-550
sudo reboot                          # reboot, then verify:
nvidia-smi                           # should list the RTX 3090

# ── 2. System packages (Python venv + tk headers are optional here) ────
sudo apt install -y python3.11 python3.11-venv python3-pip git

# ── 3. Get the code ────────────────────────────────────────────────────
cd ~
git clone https://github.com/shubham-web/qwen-image-edit-ui.git
cd qwen-image-edit-ui/server

# ── 4. Virtual environment ─────────────────────────────────────────────
python3.11 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

# ── 5. CUDA build of PyTorch (cu121 is a safe match for the 3090) ──────
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ── 6. Server + model deps (diffusers, transformers, bitsandbytes, ...) ─
pip install -r requirements.txt

# ── 7. Helpful for tight VRAM (reduces fragmentation) ──────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ── 8. Open the port so your PC can reach it ───────────────────────────
sudo ufw allow 8000/tcp

# ── 9. Run the server (4-bit precision by default) ─────────────────────
python server.py
```

> To make it survive SSH disconnects / reboots, run it as a `systemd` service
> instead — see [Run as a service (Ubuntu)](#run-as-a-service-ubuntu) below.

## Setup — Windows

Tested on **Windows 10 / 11** with an NVIDIA RTX 3090. Run these in
**PowerShell**. (Install [Python 3.11](https://www.python.org/downloads/) and
[Git](https://git-scm.com/download/win) first, and a recent NVIDIA driver so
`nvidia-smi` works.)

```powershell
cd C:\Users\pc\Desktop

git clone https://github.com/shubham-web/qwen-image-edit-ui.git
cd qwen-image-edit-ui\server

py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel

# CUDA build of PyTorch (cu121 matches the 3090; use cu128 if you prefer)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# Helpful for tight VRAM
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Open the firewall port once (run PowerShell as Administrator)
New-NetFirewallRule -DisplayName "Qwen Image API" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow

python .\server.py
```

> **bitsandbytes on Windows:** `requirements.txt` pins `bitsandbytes>=0.43.0`,
> which has native Windows wheels. If `pip` still can't install it, set
> `$env:QIE_PRECISION="bf16"` — but bf16 won't fit in 32 GB RAM, so getting
> bitsandbytes working (4-bit) is strongly recommended on this hardware.

## Run

Once installed (either OS), from the `server/` folder with the venv active:

```bash
python server.py
# or, equivalently:
uvicorn server:app --host 0.0.0.0 --port 8000
```

The model loads **in the background** on startup; `GET /health` reports
progress (`loading` → `ready`). The first request waits until it's ready.

You should see it come up on `http://0.0.0.0:8000`. From your personal PC,
point the client at `http://<server-lan-ip>:8000`.

### Run as a service (Ubuntu)

So the server auto-starts and keeps running after logout/reboot. Adjust the
paths/user, then:

```bash
sudo tee /etc/systemd/system/qwen-image-api.service >/dev/null <<'EOF'
[Unit]
Description=Qwen Image Edit API
After=network-online.target
Wants=network-online.target

[Service]
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/qwen-image-edit-ui/server
Environment=QIE_PRECISION=4bit
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ExecStart=/home/YOUR_USER/qwen-image-edit-ui/server/venv/bin/python server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now qwen-image-api
sudo systemctl status qwen-image-api      # check it's running
journalctl -u qwen-image-api -f           # follow logs (model load progress)
```

### Configuration (environment variables)

| Variable        | Default                | Description                                   |
| --------------- | ---------------------- | --------------------------------------------- |
| `QIE_MODEL_ID`  | `Qwen/Qwen-Image-Edit` | HF repo id (try `Qwen/Qwen-Image-Edit-2509`)  |
| `QIE_PRECISION` | `4bit`                 | `4bit` \| `8bit` \| `bf16`                     |
| `QIE_HOST`      | `0.0.0.0`              | Bind host                                     |
| `QIE_PORT`      | `8000`                 | Bind port                                     |
| `QIE_API_KEY`   | *(unset)*              | If set, clients must send `X-API-Key`         |

## API

### `GET /health` (also `/status`)

```json
{ "status": "ready", "device": "cuda", "precision": "4bit",
  "model_id": "Qwen/Qwen-Image-Edit", "load_seconds": 92.4, "error": null }
```

### `POST /edit` — `multipart/form-data`

| Field                 | Type  | Default | Description        |
| --------------------- | ----- | ------- | ------------------ |
| `image`               | file  | —       | Input image        |
| `prompt`              | text  | —       | Edit instruction   |
| `num_inference_steps` | int   | 30      | Denoising steps    |
| `true_cfg_scale`      | float | 4.0     | Prompt adherence   |
| `negative_prompt`     | text  | ""      | Things to avoid    |
| `seed`                | int   | 0       | Reproducibility    |

Returns the edited image as `image/png` bytes.

```bash
curl -X POST http://localhost:8000/edit \
  -F "image=@input.png" \
  -F "prompt=make him wear cool gaming headphones" \
  -F "num_inference_steps=30" \
  --output out.png
```

## Local smoke test (no HTTP)

`main.py` runs the same loader directly to confirm the model works on the box:

```bash
python main.py input.png "make him wear cool gaming headphones" output.png
```

## Opening it to your personal PC

`--host 0.0.0.0` binds all interfaces. From your PC, point the client at
`http://<server-lan-ip>:8000`. Allow the port through the server firewall:

```powershell
New-NetFirewallRule -DisplayName "Qwen Image API" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

For anything beyond your LAN, put it behind a reverse proxy (TLS) and set
`QIE_API_KEY`.

## Project layout

```
qwen_image_edit_api/
├── server/                 # runs on the GPU box
│   ├── server.py           # FastAPI app (endpoints)
│   ├── model_manager.py    # memory-efficient loading + inference
│   ├── main.py             # optional local CLI smoke test
│   └── requirements.txt
└── client/                 # runs on your personal PC
    ├── client_gui.py       # Tkinter GUI → talks to the API
    ├── run_client.py
    └── requirements.txt
```
