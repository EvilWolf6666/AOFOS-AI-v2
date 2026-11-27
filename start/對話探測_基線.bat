@echo off
rem Baseline probe for P1 step 3: empty memory, checks baseline config, 5 chained sessions (tools/chat_probe.py --baseline).
rem Close AOFOS first; keep the inference backend running. Takes about an hour.
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
uv run --locked python -m tools.chat_probe --baseline
pause
