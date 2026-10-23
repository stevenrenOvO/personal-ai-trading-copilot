@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" -Demo -Port 8001 -OpenBrowser
pause
