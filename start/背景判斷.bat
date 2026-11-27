@echo off
rem Background memory judging while AOFOS is closed (the inference backend must stay running).
rem Exits by itself when done, or as soon as AOFOS starts. Window stays open on error.
rem (ASCII only: cmd reads this file in the system code page before chcp runs)
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0.."
rem Runs in the uv environment (.venv) defined by pyproject.toml + uv.lock. No fallback to system Python:
rem missing uv or an out-of-date uv.lock stops here with an error.
where uv >nul 2>nul
if errorlevel 1 (
  echo [ERROR] uv not found in PATH. Install uv, reopen this window, then retry.
  pause
  exit /b 1
)
uv run --locked python -m tools.judge_worker
if errorlevel 1 pause

