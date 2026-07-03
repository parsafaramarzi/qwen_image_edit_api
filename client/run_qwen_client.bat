@echo off
cd /d "%~dp0"
set QIE_SSL_VERIFY=0
set QIE_SERVER_URL=https://193.93.169.217:8000
"..\.venv\Scripts\python.exe" run_client.py
pause