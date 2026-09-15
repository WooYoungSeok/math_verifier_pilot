@echo off
REM Runs the whole generation chain detached from any terminal session.
REM Every step resumes from what is already in outputs/, so re-running is safe.
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set TF_CPP_MIN_LOG_LEVEL=3
set PY=C:\Users\dssal\anaconda3\python.exe
"%PY%" scripts\run_judge.py --model math7b >> outputs\logs\math7b.log 2>&1
"%PY%" scripts\run_judge.py --model 7b >> outputs\logs\7b.log 2>&1
echo CHAIN_DONE %date% %time% >> outputs\logs\chain.log
