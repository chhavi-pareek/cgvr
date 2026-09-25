#!/bin/bash
# Push unity/Parity3D/Assets into the real Unity project and compile it headlessly.
#
#   tools/sync_unity.sh                 # sync + compile, print errors
#   tools/sync_unity.sh --no-compile    # sync only
#   tools/sync_unity.sh --smoke         # sync + compile + run the headless sim smoke test
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

echo "sync  $REPO/unity/Parity3D/Assets/  ->  $PROJ/Assets/"
rsync -a --delete \
      "$REPO/unity/Parity3D/Assets/Scripts/" "$PROJ/Assets/Scripts/"
rsync -a --delete \
      "$REPO/unity/Parity3D/Assets/Shaders/" "$PROJ/Assets/Shaders/"
rsync -a --delete \
      "$REPO/unity/Parity3D/Assets/Editor/" "$PROJ/Assets/Editor/"

[ "${1:-}" = "--no-compile" ] && { echo "synced; the open editor recompiles when it regains focus"; exit 0; }

# Only the compile needs the project lock; syncing into a live editor is the fast path, because
# Unity hot-reloads the changed scripts as soon as the window regains focus.
if pgrep -x Unity >/dev/null 2>&1; then
    echo "synced, but the editor is running so batchmode cannot take the project lock."
    echo "Either quit Unity (Cmd+Q) and rerun, or just click back into Unity to hot-reload."
    exit 2
fi

mkdir -p "$PROJ/Logs"
EXTRA=""
if [ "${1:-}" = "--smoke" ]; then EXTRA="-executeMethod ParitySmoke.Run"; fi
echo "compiling headlessly${EXTRA:+ + smoke test} (first run imports everything, so it is slow) ..."
"$UNITY" -batchmode -quit -nographics \
         -projectPath "$PROJ" $EXTRA \
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
if [ "${1:-}" = "--smoke" ]; then
    grep -F "[ParitySmoke]" "$LOG" | sed 's/^/  /'
fi
echo "compiled clean"
