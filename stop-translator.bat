@echo off
REM ── Stop the Shiori translation server ──────────────────────────────────────
REM Frees ports 5003 (API) and 5004 (worker) by stopping whatever is listening on
REM them. Use this if the server was left running without a window to close.
powershell -NoProfile -Command "$any=$false; foreach ($port in 5003,5004) { Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue; $script:any=$true } }; if ($any) { Write-Host 'Translation server stopped.' } else { Write-Host 'Server was not running.' }"
timeout /t 2 >nul
