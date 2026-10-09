@echo off
rem Manual review: yes = approve, no = reject, Enter = skip. Closes when done.
rem (ASCII only: cmd reads this file in the system code page before chcp runs)
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0.."
if exist "venv\Scripts\python.exe" (set "PY=venv\Scripts\python.exe") else (set "PY=python")
"%PY%" -m tools.memory_review
if errorlevel 1 pause
