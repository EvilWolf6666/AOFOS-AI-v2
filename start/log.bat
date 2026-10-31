@echo off
rem Live view of today's log (logs\daily\YYYY-MM-DD.log). The start script opens it; double-click to reopen.
rem A running AOFOS keeps writing the file of the day it started, so this window keeps up past midnight.
rem (ASCII only: cmd reads this file in the system code page before chcp runs)
chcp 65001 >nul
cd /d "%~dp0.."
title AOFOS log
powershell -NoProfile -Command "[Console]::OutputEncoding=[Text.Encoding]::UTF8; $f='logs\daily\'+(Get-Date -Format yyyy-MM-dd)+'.log'; while(-not(Test-Path $f)){Start-Sleep 1}; Get-Content $f -Wait -Tail 50 -Encoding UTF8"
