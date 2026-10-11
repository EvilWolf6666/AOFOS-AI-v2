@echo off
rem Background memory judging while AOFOS is closed (llama-server must stay running).
rem Exits by itself when done, or as soon as AOFOS starts. Window stays open on error.
rem (ASCII only: cmd reads this file in the system code page before chcp runs)
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0.."
if exist "venv\Scripts\python.exe" (set "PY=venv\Scripts\python.exe") else (set "PY=python")
"%PY%" -m tools.judge_worker
if errorlevel 1 pause

