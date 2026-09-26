#!/bin/bash
# 큐에서 한 줄씩 꺼내 벤치 작업을 돈다. GPU 한 장당 하나씩 띄운다.
# 큐 한 줄: "<조합> <시드>"
W=${AF_WORK:-/root/work}
GPU=$1; Q=$2; KIND=${3:-pg}
while true; do
  LINE=$(flock $Q.lock -c "head -1 $Q; sed -i 1d $Q")
  [ -z "$LINE" ] && break
  set -- $LINE; C=$1; S=$2
  AF_WORK=$W AF_CONDA=${AF_CONDA:-/opt/conda/bin} \
    bash $W/anchorflow/bench_${KIND}.sh $C $S-$S $GPU
  echo "[$(date +%H:%M:%S)] gpu$GPU $KIND $C s$S 완료" >> $W/bench_progress.log
done
echo "WORKER_DONE gpu$GPU $KIND" >> $W/bench_progress.log
