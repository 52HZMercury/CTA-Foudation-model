#!/bin/bash
# 用法: bash run_train.sh --grid merlin --fold 0 [其余 train.py 参数]
#   E0: bash run_train.sh --grid merlin --fold 0
#   E2: bash run_train.sh --grid merlin --fold 0 --arch merlin --adabn
#
# 每次启动：
#   logs/<YYYYmmdd_HHMMSS>/   ← 权重 best/last.pth、log.csv、TensorBoard
#   out.log                   ← 本次 stdout（覆盖上一次），终端 tail 实时看
#
# 训练用 setsid nohup 后台跑；Ctrl-C 只停 tail，不停训练。

set -u
cd "$(dirname "$0")" || exit 1
PY=${PY:-$HOME/anaconda3/envs/ctafdm/bin/python}

TS=$(date +%Y%m%d_%H%M%S)
OUT=logs/$TS
mkdir -p "$OUT"

setsid nohup "$PY" -u -m valve_train.train "$@" --out "$OUT" > out.log 2>&1 < /dev/null &
PID=$!
echo "== run  $OUT  (pid $PID)"
echo "== log  $(pwd)/out.log   权重 $OUT"
echo "== Ctrl-C 只退出预览，训练继续；再看: tail -f out.log"
tail -f --pid="$PID" out.log
wait "$PID"
rc=$?
echo "== exit $rc -> $OUT"
exit $rc
