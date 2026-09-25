#!/bin/bash
# Push unity/Parity3D/Assets into the real Unity project and compile it headlessly.
#
#   tools/sync_unity.sh                 # sync + compile, print errors
#   tools/sync_unity.sh --no-compile    # sync only
#
# Batchmode compilation is the whole point: the editor does not have to be open, and script
# errors come back here instead of having to be read off the Console by hand.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJ="${PARITY_UNITY_PROJECT:-/Users/user/UnityProjects/Parity3D}"
UNITY="${PARITY_UNITY_BIN:-/Applications/Unity/Hub/Editor/6000.6.3f1/Unity.app/Contents/MacOS/Unity}"
LOG="$PROJ/Logs/batch_compile.log"

[ -d "$PROJ" ] || { echo "no project at $PROJ (set PARITY_UNITY_PROJECT)"; exit 1; }
[ -x "$UNITY" ] || { echo "no editor at $UNITY (set PARITY_UNITY_BIN)"; exit 1; }

if pgrep -f "Unity.app/Contents/MacOS/Unity" >/dev/null 2>&1; then
    echo "The Unity editor is running. Batchmode cannot open a project the editor holds a"
    echo "lock on, so quit Unity (Cmd+Q) and run this again."
    exit 1
fi

echo "sync  $REPO/unity/Parity3D/Assets/  ->  $PROJ/Assets/"
rsync -a --delete \
      "$REPO/unity/Parity3D/Assets/Scripts/" "$PROJ/Assets/Scripts/"
rsync -a --delete \
      "$REPO/unity/Parity3D/Assets/Shaders/" "$PROJ/Assets/Shaders/"

[ "${1:-}" = "--no-compile" ] && { echo "synced (compile skipped)"; exit 0; }

mkdir -p "$PROJ/Logs"
echo "compiling headlessly (first run imports the whole project, so it is slow) ..."
"$UNITY" -batchmode -quit -nographics \
         -projectPath "$PROJ" \
         -logFile "$LOG" >/dev/null 2>&1
rc=$?

# Unity reports script errors in the log rather than in the exit code, so scan for both.
errs=$(grep -E "error CS[0-9]+|Shader error|error:" "$LOG" 2>/dev/null | sort -u)
if [ -n "$errs" ]; then
    echo "--- COMPILE ERRORS ---"
    echo "$errs" | head -60
    exit 1
fi
if [ $rc -ne 0 ]; then
    echo "--- unity exited $rc with no CS errors; tail of $LOG ---"
    tail -25 "$LOG"
    exit $rc
fi
echo "compiled clean"
