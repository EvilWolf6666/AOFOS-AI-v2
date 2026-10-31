@echo off
rem Scripted chat probe against the inference backend (tools/chat_probe.py). Uses a COPY of data/memory.db.
rem Close AOFOS first; keep the inference backend running.
rem (ASCII only: cmd reads this file in the system code page before chcp runs)
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0.."
if exist "venv\Scripts\python.exe" (set "PY=venv\Scripts\python.exe") else (set "PY=python")
"%PY%" -m tools.chat_probe
pause
