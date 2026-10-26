@echo off
rem Baseline probe for P1 step 3: empty memory, checks baseline config, 5 chained sessions (tools/chat_probe.py --baseline).
rem Close AOFOS first; keep the inference backend running. Takes about an hour.
rem (ASCII only: cmd reads this file in the system code page before chcp runs)
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0.."
if exist "venv\Scripts\python.exe" (set "PY=venv\Scripts\python.exe") else (set "PY=python")
"%PY%" -m tools.chat_probe --baseline
pause
