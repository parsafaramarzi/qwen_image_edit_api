# 🖼️ Qwen Image Editor — Client

Lightweight desktop GUI that runs on **your personal PC**. It has no model,
no GPU dependency, and no `torch`/`diffusers` — it just sends images to the
[server](../server) over HTTP and shows the result.

## Install

```bash
cd client
pip install -r requirements.txt
```

## Run

```bash
python run_client.py
# or
python client_gui.py
```

## Connect to your server

Set the server URL either in the **Server → URL** field in the UI, or via an
environment variable before launching:

```bash
# Windows (PowerShell)
$env:QIE_SERVER_URL = "http://192.168.1.50:8000"
python run_client.py

# macOS / Linux
export QIE_SERVER_URL="http://192.168.1.50:8000"
python run_client.py
```

If the server was started with `QIE_API_KEY`, put the same key in the
**API Key** field (or set `QIE_API_KEY` in the client environment).

Click **🔌 Check** to confirm the server is reachable and the model is loaded
(🟢 Ready). Then browse an image, enter a prompt, and click **🎨 Edit Image**.

The first edit after the server boots may be slow while the model warms up.
