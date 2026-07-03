@echo off
REM ── Shiori translation server launcher ──────────────────────────────────────
REM Starts manga-image-translator's API on http://127.0.0.1:5003 (GPU-accelerated)
REM and auto-spawns its translator worker on port 5004. Set this same URL in
REM Shiori → Settings → Translation. First launch downloads the AI models (~a few GB).
REM Leave this window open while translating; close it (or Ctrl+C) to stop.

cd /d "%~dp0"
REM Disable the internal API<->worker nonce. The experimental server never forwards
REM the X-Nonce header when dispatching to its worker, so leaving it on causes a 401
REM "Nonce does not match". The worker only listens on localhost, so this is safe.
set "MT_WEB_NONCE=None"
REM Force UTF-8 so the LLM translators don't crash logging Japanese on the Windows console.
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
title Shiori Translation Server (manga-image-translator)
echo Starting translation server on http://127.0.0.1:5003 ...
echo (First run downloads models - this can take several minutes.)
echo.
REM No --verbose: that flag saves every intermediate step image + final.png into result\,
REM which balloons disk. The streaming API returns the image directly and saves nothing.
REM --models-ttl 300: unload models from VRAM after 5 minutes idle (frees GPU for gaming).
"%~dp0.venv\Scripts\python.exe" server\main.py --host 127.0.0.1 --port 5003 --use-gpu --context-size 4 --models-ttl 300
echo.
echo Server stopped. Press any key to close.
pause >nul
