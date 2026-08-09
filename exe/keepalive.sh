#!/bin/bash
# Keep a training run alive on a vast.ai instance until it finishes.
#
# Two things kill these runs and neither is a crash. The idle watchdog stops an
# instance five minutes after the GPU goes quiet, which is correct behaviour and
# has stopped instances the moment a run completed -- but it has also caught
# them during a gap. And the container's sshd has come back unreachable after a
# restart, which looks identical to a dead job from outside.
#
# So: every INTERVAL, make sure the instance is running, make sure the training
# process is alive, and if it is not, work out whether the run finished or died.
# A finished run is left alone. A dead one is relaunched with --resume, which
# picks up the model, the optimiser, the RNG and the DAgger pool from the last
# checkpoint.
#
#   keepalive.sh <instance_id> <ssh_host> <ssh_port> <name> <tmux_session> <cmd...>
#
# The command is the original launch line minus --resume; this adds it.
set -u
ID=$1; HOST=$2; PORT=$3; NAME=$4; SESS=$5; shift 5
CMD="$*"
INTERVAL=${KEEPALIVE_INTERVAL:-180}
LOG=/tmp/keepalive_$NAME.log

say() { echo "[$(date +%H:%M:%S)] $*" >> $LOG; }
rsh() { timeout 90 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
        -o BatchMode=yes -p $PORT root@$HOST "$@" 2>/dev/null; }

say "watching $NAME on $ID ($HOST:$PORT), checking every ${INTERVAL}s"

while true; do
  STATE=$(timeout 60 vastai show instances-v1 --raw 2>/dev/null | python3 -c "
import json,sys
for i in json.load(sys.stdin)['instances']:
    if i['id']==$ID: print(i.get('actual_status') or 'unknown')
" 2>/dev/null)

  if [ -z "$STATE" ]; then
    say "instance $ID not in the list any more; giving up"
    exit 1
  fi

  if [ "$STATE" != "running" ]; then
    say "instance is '$STATE'; starting it"
    timeout 60 vastai start instance $ID >/dev/null 2>&1
    sleep 90
    continue
  fi

  if ! rsh 'echo ok' | grep -q ok; then
    say "instance says running but ssh refused; waiting"
    sleep 60
    continue
  fi

  if rsh "ps aux | grep -q '[t]rain_nextstate.*$NAME'" ; then
    PROG=$(rsh "tr '\r' '\n' < /workspace/$NAME.log | grep -oE '[0-9]+/[0-9]+ \[' | tail -1")
    say "alive, $PROG"
  elif rsh "grep -q '^\[rollout\] mean over' /workspace/$NAME.log"; then
    say "finished; backing up and stopping"
    rsh "rclone copy /workspace/$NAME.log r2:storage/result/anchorflow/log/ ;
         rclone copy /workspace/$NAME.pt r2:storage/result/anchorflow/ckpt/$NAME/"
    exit 0
  else
    IT=$(rsh "python3 -c \"
import torch
try: print(torch.load('/workspace/$NAME.pt', map_location='cpu', weights_only=False)['iter'])
except Exception: print(0)\"")
    say "process gone at iteration ${IT:-0}; relaunching with --resume"
    rsh "cd /workspace/anchorflow && git pull -q
         tmux kill-session -t $SESS 2>/dev/null
         tmux new-session -d -s $SESS \"cd /workspace/SC-GS && export PYTHONPATH=/workspace/anchorflow/lib:/workspace/SC-GS && $CMD --resume /workspace/$NAME.pt >> /workspace/$NAME.log 2>&1\""
    sleep 60
  fi

  sleep $INTERVAL
done
