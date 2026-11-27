#!/bin/bash
# 用法: bash run_ssl.sh [其余 ssl.py 参数]
#   默认 = E3c：SimSiam，100 epoch，enc lr 1e-5（§12.7/§12.8 的最优配置）
#   历史对照：bash run_ssl.sh --objective mask --epochs 30
#
# 每次启动：
#   logs/<YYYYmmdd_HHMMSS>/   ← encoder.pth、log.csv
#   out.log                   ← 本次 stdout（覆盖上一次），终端 tail 实时看
# 产物 encoder.pth 给下游微调：
#   bash run_train.sh --grid merlin --fold N --arch merlin --adabn --merlin-ckpt logs/<TS>/encoder.pth
#
# 训练用 setsid nohup 后台跑；Ctrl-C 只停 tail，不停训练。

set -u
cd "$(dirname "$0")" || exit 1
PY=${PY:-$HOME/anaconda3/envs/ctafdm/bin/python}

TS=$(date +%Y%m%d_%H%M%S)
OUT=logs/$TS
mkdir -p "$OUT"

setsid nohup "$PY" -u -m valve_train.ssl "$@" --out "$OUT" > out.log 2>&1 < /dev/null &
PID=$!
echo "== ssl  $OUT  (pid $PID)"
echo "== log  $(pwd)/out.log   权重 $OUT"
echo "== Ctrl-C 只退出预览，训练继续；再看: tail -f out.log"
tail -f --pid="$PID" out.log
wait "$PID"
rc=$?
echo "== exit $rc -> $OUT"
exit $rc
