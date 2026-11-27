@echo off
rem Run the v2 question set against the model in config.LLAMA_SERVER (tools/eval_runner.py).
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
uv run --locked python -m tools.eval_runner run
pause
