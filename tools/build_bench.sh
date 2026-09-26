#!/bin/bash
# Build the standalone frame bench and zip it for another machine.
#
#   tools/build_bench.sh [win64|mac]      -> <project>/Builds/ParityBench-<target>.zip
#
# Needs the target's Unity build support module (Unity Hub > Installs > Add modules) and the
# editor closed (batchmode takes the project lock).
set -eu
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJ="${PARITY_UNITY_PROJECT:-/Users/user/UnityProjects/Parity3D}"
UNITY="${PARITY_UNITY_BIN:-/Applications/Unity/Hub/Editor/6000.6.3f1/Unity.app/Contents/MacOS/Unity}"
T="${1:-win64}"
OUT="$PROJ/Builds/$T"
bash "$REPO/tools/sync_unity.sh" --no-compile >/dev/null
echo "building $T ..."
if ! "$UNITY" -batchmode -quit -projectPath "$PROJ" -executeMethod ParityBuild.Bench -target "$T" -out "$OUT" \
     -logFile "$PROJ/Logs/build_$T.log"; then
  grep -E "error CS|\[ParityBuild\]|Error" "$PROJ/Logs/build_$T.log" | head -20
  exit 1
fi
grep "\[ParityBuild\]" "$PROJ/Logs/build_$T.log" | head -3
if [ "$T" = mac ]; then cp "$REPO/tools/bench_package/run_bench.sh" "$OUT/"; else sed 's/\r*$/\r/' "$REPO/tools/bench_package/run_bench.bat" > "$OUT/run_bench.bat"; fi
rm -f "$PROJ/Builds/ParityBench-$T.zip"
(cd "$PROJ/Builds" && zip -qr "ParityBench-$T.zip" "$T")
ls -lh "$PROJ/Builds/ParityBench-$T.zip" | awk '{print $5, $9}'
