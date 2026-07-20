@echo off
REM ── Shiori translation server — auto-restart wrapper (remote hosting) ────────
REM Same server as start-translator.bat, but restarts it automatically if it
REM crashes, so the remote endpoint stays up unattended. Restarts are logged to
REM logs\server-restarts.log. Close this window (or run stop-translator.bat)
REM to stop it for real.
REM Remote config (access token, allowed origins, limits) lives in .env —
REM see REMOTE-SETUP.md.

cd /d "%~dp0"
REM Same required env as start-translator.bat (see the comments there).
set "MT_WEB_NONCE=None"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
title Shiori Translation Server (auto-restart)
if not exist logs mkdir logs

REM Any arguments passed to this .bat are forwarded to main.py (e.g. --verbose,
REM or overriding --port). They come after the defaults below, so they win.
:loop
echo [%date% %time%] starting server >> logs\server-restarts.log
"%~dp0venv\Scripts\python.exe" server\main.py --host 127.0.0.1 --port 5003 --use-gpu --context-size 4 --models-ttl 300 %*
echo [%date% %time%] server exited (code %errorlevel%) - restarting in 5s >> logs\server-restarts.log
echo Server exited - restarting in 5s (Ctrl+C or close the window to stop)...
timeout /t 5 /nobreak >nul
goto loop
