# 🎨 Qwen Image Studio — Web Chat Client

A ChatGPT/Gemini-style web UI for the Qwen Image Edit/Generate server. Runs on
your PC in the browser; the model runs on the server. This is an **alternative**
to the Tkinter client in [`../client`](../client) — both talk to the same server.

## Features
- **Chat interface** — a prompt box at the bottom, message history above.
- **Drag & drop / browse** images to attach as input (native to Streamlit).
- **Multiple chats**, saved locally and cached (`~/.qwen_chat_ui/`), with a
  **New chat** button and per-chat delete.
- **Use output as input** — click **↩ Use as input** under any generated image
  to feed it into your next prompt, no manual saving.
- **Model + LoRA + settings** in the sidebar: pick a preset or a custom repo,
  choose precision, load/unload the model and LoRA, tune steps/CFG/seed/etc.
- **Editing *and* generation** — attach an image to edit it, or leave it empty
  to generate from a text prompt (if a generation model is loaded). Attaching an
  image to a text-to-image model is harmless (the image is ignored).

## Install & run

```powershell
cd Client_Chat_UI
pip install -r requirements.txt
streamlit run chat_app.py
```

Or double-click **`run_chat_ui.bat`** (it uses the shared `../.venv` and sets
`QIE_SSL_VERIFY=0` for the self-signed server cert). Streamlit opens the app in
your browser (default http://localhost:8501).

## Configuration (env vars)
| Variable          | Default                        | Description                          |
| ----------------- | ------------------------------ | ------------------------------------ |
| `QIE_SERVER_URL`  | `https://193.93.169.217:8000`  | Server address                       |
| `QIE_API_KEY`     | *(empty)*                      | If the server requires an API key    |
| `QIE_SSL_VERIFY`  | `1`                            | Set `0` for the self-signed TLS cert |

You can also edit the URL/API key live in the sidebar under **🌐 Server**.

## Typical flow
1. Sidebar → pick a model → **⬇️ Load** (wait for 🟢 Ready).
2. (Optional) attach images via drag-and-drop.
3. Type your prompt in the chat box → send.
4. When the image comes back, click **↩ Use as input** to iterate on it.

Chats and images are stored under `~/.qwen_chat_ui/` so they persist across runs.
