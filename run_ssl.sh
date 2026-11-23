#!/bin/bash
# 用法: bash run_ssl.sh --out work/ssl/e3 [其余 ssl.py 参数]
#   E3 自监督（一次性，与折无关）：100 iter/epoch × 30 epoch
# 产物: <out>/encoder.pth → 下游微调：
#   bash run_train.sh --grid merlin --fold N --arch merlin --adabn --merlin-ckpt <out>/encoder.pth
#
# 训练用 setsid nohup 后台跑；Ctrl-C 只停 tail，不停训练。

set -u
cd "$(dirname "$0")" || exit 1
PY=${PY:-$HOME/anaconda3/envs/ctafdm/bin/python}

setsid nohup "$PY" -u -m valve_train.ssl "$@" > out_ssl.log 2>&1 < /dev/null &
PID=$!
echo "== ssl  (pid $PID)  log $(pwd)/out_ssl.log"
echo "== Ctrl-C 只退出预览，训练继续；再看: tail -f out_ssl.log"
tail -f --pid="$PID" out_ssl.log
wait "$PID"
rc=$?
echo "== exit $rc"
exit $rc
