@echo off
rem Run the v2 question set against the model in config.LLAMA_SERVER (tools/eval_runner.py).
rem (ASCII only: cmd reads this file in the system code page before chcp runs)
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0.."
if exist "venv\Scripts\python.exe" (set "PY=venv\Scripts\python.exe") else (set "PY=python")
"%PY%" -m tools.eval_runner run
pause
