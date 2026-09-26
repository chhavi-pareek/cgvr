#!/bin/bash
# PARITY frame bench, standalone Mac build (the Windows build uses run_bench.bat). Same
# protocol and modes: ./run_bench.sh [quick|window]
cd "$(dirname "$0")"
mkdir -p Results
BIN=./ParityBench.app/Contents/MacOS/ParityBench
MODE=-batchmode; [ "${1:-}" = window ] && MODE=
COMMON="-parityBench -overlap -sizes 8000 -out Results"
if [ "${1:-}" = quick ]; then
  "$BIN" $MODE $COMMON -seed 1 -targets 10,14 -policies masslod,knapsack_fullsim,parity -tag quick -logFile Results/quick.log
else
  for k in 1 2 3; do
    echo "seed $k of 3 ..."
    "$BIN" $MODE $COMMON -seed $k -targets 8,10,12,14,17,20 -policies masslod,timeslice,knapsack,knapsack_fullsim,parity,parity_sw0 -tag seed$k -logFile Results/seed$k.log
    [ $k -lt 3 ] && { echo "cooling down for 3 minutes ..."; sleep 180; }
  done
fi
echo "done: results are in $PWD/Results"
