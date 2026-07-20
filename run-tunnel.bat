@echo off
REM ── Cloudflare named tunnel — auto-restart wrapper ───────────────────────────
REM Exposes the local server (http://127.0.0.1:5003) as your HTTPS hostname via
REM the named tunnel configured in %USERPROFILE%\.cloudflared\config.yml.
REM One-time setup (login, tunnel create, DNS route): see REMOTE-SETUP.md.
REM cloudflared reconnects on its own for network blips; this loop only covers
REM the process itself dying.

set "TUNNEL_NAME=shiori-translate"
title Shiori Translate Tunnel (cloudflared)

:loop
cloudflared tunnel run %TUNNEL_NAME%
echo Tunnel exited (code %errorlevel%) - restarting in 5s (Ctrl+C or close the window to stop)...
timeout /t 5 /nobreak >nul
goto loop
