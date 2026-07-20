@echo off
REM Removes the Startup-folder launcher installed by install-autostart.bat.
set "SP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\shiori-translate.bat"
if exist "%SP%" (
    del "%SP%"
    echo Removed %SP%
) else (
    echo Nothing to remove - %SP% not found.
)
pause
