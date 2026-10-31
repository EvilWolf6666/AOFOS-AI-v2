@echo off
rem Same as the other bat, but WITHOUT the persona system prompt (C22 test, tools/eval_runner.py --no-persona).
rem (ASCII only: cmd reads this file in the system code page before chcp runs)
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0.."
if exist "venv\Scripts\python.exe" (set "PY=venv\Scripts\python.exe") else (set "PY=python")
"%PY%" -m tools.eval_runner run --no-persona
pause
