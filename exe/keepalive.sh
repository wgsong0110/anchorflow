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
#
# PATTERN and DONE_MARK say what to look for; they default to the trainer. Any
# script with a --resume and a line it prints when it finishes can use this --
# the fit lost forty minutes to a host reclaiming its GPU, which is the same
# failure this already handles for training.
set -u
ID=$1; HOST=$2; PORT=$3; NAME=$4; SESS=$5; shift 5
CMD="$*"
RESUME_ARG=${RESUME_ARG:-"--resume /workspace/$4.pt"}
INTERVAL=${KEEPALIVE_INTERVAL:-180}
PATTERN=${PATTERN:-train_nextstate}
QUEUED=0
QUEUE_LIMIT=${QUEUE_LIMIT:-5}
DONE_MARK=${DONE_MARK:-'^\[rollout\] mean over'}
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

  if [ "$STATE" = "scheduling" ]; then
    # vast.ai has taken the contract and has no host for it. This does not
    # resolve on a timetable and the disk is billed while it sits there
    say "instance is scheduling: no host. Destroying rather than waiting."
    timeout 60 vastai destroy instance $ID -y >/dev/null 2>&1
    exit 2
  fi

  if [ "$STATE" != "running" ]; then
    say "instance is '$STATE'; starting it"
    OUT=$(timeout 60 vastai start instance $ID 2>&1)
    if echo "$OUT" | grep -qi "resources are currently unavailable\|state change queued"; then
      # This message has twice resolved on its own within minutes -- the host
      # simply had no room at that instant. Destroying on the first sighting
      # threw away a running fit. It counts as evidence only when it keeps
      # saying so, and when the instance is still not running afterwards
      QUEUED=$((QUEUED + 1))
      say "start says the host has no room (${QUEUED}/${QUEUE_LIMIT})"
      if [ $QUEUED -ge $QUEUE_LIMIT ]; then
        say "queued $QUEUED times over $((QUEUE_LIMIT * 3)) minutes; destroying"
        timeout 60 vastai destroy instance $ID -y >/dev/null 2>&1
        exit 2
      fi
    else
      QUEUED=0
    fi
    sleep 90
    continue
  fi

  if ! rsh 'echo ok' | grep -q ok; then
    say "instance says running but ssh refused; waiting"
    sleep 60
    continue
  fi

  if rsh "ps aux | grep -q '[${PATTERN:0:1}]${PATTERN:1}.*$NAME'" ; then
    PROG=$(rsh "tr '\r' '\n' < /workspace/$NAME.log | grep -oE '[0-9]+/[0-9]+ \[' | tail -1")
    say "alive, $PROG"
  elif rsh "grep -q '$DONE_MARK' /workspace/$NAME.log"; then
    say "finished; backing up and stopping"
    rsh "rclone copy /workspace/$NAME.log r2:storage/result/anchorflow/log/ ;
         rclone copy /workspace/$NAME.pt r2:storage/result/anchorflow/ckpt/$NAME/"
    exit 0
  else
    IT=$(rsh "python3 -c \"
import torch
for f in ('/workspace/$NAME.pt.state', '/workspace/$NAME.pt'):
    try:
        print(torch.load(f, map_location='cpu', weights_only=False)['iter']); break
    except Exception: pass
else: print(0)\"")
    say "process gone at iteration ${IT:-0}; relaunching with --resume"
    rsh "cd /workspace/anchorflow && git pull -q
         tmux kill-session -t $SESS 2>/dev/null
         tmux new-session -d -s $SESS \"cd /workspace/SC-GS && export PYTHONPATH=/workspace/anchorflow/lib:/workspace/SC-GS && $CMD $RESUME_ARG >> /workspace/$NAME.log 2>&1\""
    sleep 60
  fi

  sleep $INTERVAL
done
