@echo off
rem PARITY frame bench, standalone: no Unity needed. Run from this folder (double-click works).
rem
rem   run_bench.bat           full protocol: 3 seeds x 33 cells at N = 8000, 3-minute cooldown
rem                           between seeds (about 40 minutes)
rem   run_bench.bat quick     1 seed, 9 cells: a check that the build runs here (about 5 minutes)
rem   run_bench.bat window    as the full protocol, in a window instead of -batchmode (use this
rem                           if Results\*.txt reports "no graphics device")
rem
rem Results\frame_bench_<tag>.csv  one row per cell     (copy these back for analysis)
rem Results\frame_bench_<tag>.txt  the readable log, starting with the machine it ran on
rem Results\<tag>.log              Unity's full player log
rem Close other programs first; keep the machine on mains power.
setlocal
cd /d "%~dp0"
if not exist Results mkdir Results
set MODE=-batchmode
if /i "%1"=="window" set MODE=
set COMMON=-parityBench -overlap -sizes 8000 -out Results
if /i "%1"=="quick" (
  echo quick check ...
  start "" /wait ParityBench.exe %MODE% %COMMON% -seed 1 -targets 10,14 -policies masslod,knapsack_fullsim,parity -tag quick -logFile Results\quick.log
  goto done
)
for %%k in (1 2 3) do (
  echo seed %%k of 3 ...
  start "" /wait ParityBench.exe %MODE% %COMMON% -seed %%k -targets 8,10,12,14,17,20 -policies masslod,timeslice,knapsack,knapsack_fullsim,parity,parity_sw0 -tag seed%%k -logFile Results\seed%%k.log
  if %%k LSS 3 (
    echo cooling down for 3 minutes ...
    timeout /t 180 /nobreak >nul
  )
)
:done
echo done: results are in %cd%\Results
pause
