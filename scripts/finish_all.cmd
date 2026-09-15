@echo off
REM Waits for run_all.cmd (outputs\logs\chain.log) and then finishes the KT
REM pipeline and the MathEdu check with the local Qwen2.5-7B judge.
REM Detached like run_all.cmd; every step is resumable, so re-running is safe.
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set TF_CPP_MIN_LOG_LEVEL=3
set PY=C:\Users\dssal\anaconda3\python.exe
set LOG=outputs\logs\finish_all.log

:wait
if not exist outputs\logs\chain.log (
    timeout /t 120 /nobreak >nul
    goto wait
)
echo START %date% %time% >> %LOG%
"%PY%" scripts\extract_labels.py --model 7b --level theme  >> %LOG% 2>&1
"%PY%" scripts\join_gen_labels.py                          >> %LOG% 2>&1
"%PY%" scripts\merge_outputs.py                            >> %LOG% 2>&1
"%PY%" scripts\run_judge.py --dataset mathedu --model 7b   >> %LOG% 2>&1
"%PY%" scripts\extract_labels.py --dataset mathedu         >> %LOG% 2>&1
"%PY%" scripts\eval_mathedu.py                             >> %LOG% 2>&1
echo ALL_DONE %date% %time% >> %LOG%
