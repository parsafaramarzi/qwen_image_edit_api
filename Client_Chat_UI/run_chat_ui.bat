@echo off
cd /d "%~dp0"
set QIE_SSL_VERIFY=0
REM Uses the venv one level up (shared with the tkinter client). Adjust if needed.
"..\.venv\Scripts\python.exe" -m streamlit run chat_app.py
pause
