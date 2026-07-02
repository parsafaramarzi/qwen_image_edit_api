# 🖼️ Qwen Image Edit — Client / Server

Run the [Qwen Image Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) model on
a GPU server and drive it from a lightweight desktop client on your own PC.

- **[`server/`](server/)** — FastAPI inference server. Runs on the GPU box
  (e.g. RTX 3090). Loads the model in a memory-efficient way (4-bit by default)
  and exposes an HTTP API. **[Setup & docs →](server/README.md)**
- **[`client/`](client/)** — Tkinter GUI. Runs on your personal PC, no GPU or
  `torch` needed; talks to the server over HTTP.
  **[Setup & docs →](client/README.md)**

## Quick start

**On the GPU server** (Ubuntu recommended):

```bash
git clone <this-repo-url>
cd qwen_image_edit_api/server
python3.11 -m venv venv && source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python server.py            # serves on http://0.0.0.0:8000
```

**On your PC**:

```bash
cd qwen_image_edit_api/client
pip install -r requirements.txt
python run_client.py        # set Server URL to http://<server-ip>:8000
```

See [`server/README.md`](server/README.md) for full Ubuntu/Windows setup,
the memory-fit explanation, `systemd` service, and API reference.
