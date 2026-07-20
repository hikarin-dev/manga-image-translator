@echo off
REM ── Autostart at logon (no admin needed) ─────────────────────────────────────
REM Drops a launcher into the Startup folder that starts the translation server
REM and the Cloudflare tunnel, minimized, whenever you log in.
REM To undo: delete the file it prints at the end (or run uninstall-autostart.bat).

set "SP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\shiori-translate.bat"
> "%SP%" echo @echo off
>>"%SP%" echo start "" /min cmd /c ""%~dp0run-remote-server.bat""
>>"%SP%" echo start "" /min cmd /c ""%~dp0run-tunnel.bat""
echo Installed autostart launcher:
echo   %SP%
echo Server + tunnel will start minimized at every logon.
pause
