@echo off
REM ── Shiori auxiliary translation node ───────────────────────────────────────
REM Runs THIS machine's GPU as extra capacity for someone else's Shiori
REM translation server. It dials out to that server and waits for work, so this
REM machine needs no tunnel, no port forward, and no inbound firewall rule.
REM Nothing here listens on a public interface.
REM
REM Fill in the two values below (the server's operator gives you both), then
REM just run this file. Leave the window open; close it to stop.
REM
REM The console stays quiet: one line per chunk. Nothing you translate for the
REM server is written to this disk: no images, no logs with its text. The
REM worker's output is kept in memory and shown here only if it fails. Add
REM --verbose to the command below to watch the worker live (still never saved).

cd /d "%~dp0"

set "MT_AUX_JOIN=https://translate.EXAMPLE.com"
set "MT_AUX_TOKEN="

REM Force UTF-8 so the translators don't crash logging Japanese on the console.
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
title Shiori Aux Node

if "%MT_AUX_TOKEN%"=="" (
  echo Set MT_AUX_TOKEN in this file first - the server operator gives you the join secret.
  pause
  exit /b 1
)

REM Restarts on crash. The node also reconnects on its own when the main server
REM restarts or the link drops, so this loop only covers the process dying.
:loop
"%~dp0venv\Scripts\python.exe" server\main.py --aux --use-gpu --models-ttl 300
if %errorlevel%==1 (
  echo.
  echo Node exited with a fatal error - check the message above ^(wrong token,
  echo version mismatch, or the local worker failed to start^). Not restarting.
  pause
  exit /b 1
)
echo Node exited (code %errorlevel%) - restarting in 5s (close the window to stop)...
timeout /t 5 /nobreak >nul
goto loop
