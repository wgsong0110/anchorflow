#!/bin/bash
# 인스턴스 산출물을 R2 로 **계속** 올린다. 언제 destroy 돼도 잃는 것이 최소가 되게.
# 사용: r2_sync.sh <이름> [주기초]
W=${AF_WORK:-/root/work}
NAME=${1:?이름을 주라}; EVERY=${2:-300}
R=r2:storage/result/anchorflow
cd $W || exit 1
while true; do
  # 1) 학습: 체크포인트(풀 상태·누적값·난수 포함) + 로그 + TB
  for d in abl_*; do
    [ -d "$d" ] && rclone copy "$d" $R/pool/$d --transfers 4 --checkers 8 >/dev/null 2>&1
  done
  for f in abl_*.log; do
    [ -f "$f" ] && rclone copy "$f" $R/pool/ >/dev/null 2>&1
  done
  [ -d tb ] && rclone copy tb $R/tb_$NAME --transfers 4 >/dev/null 2>&1
  # 2) 벤치: 시나리오·덤프·지표·로그. h5dir 은 i-PG 가 프레임마다 쓰는 중간물이라
  #    같이 올려 둔다 -- destroy 돼도 끝난 프레임은 살린다.
  for d in bench bench_cfg bench_cfg_f60 bench_pg bench_gsv bench_stu \
           bench_ipg20f60 bench_ipg8f60 bench_met gaussim_data work_dirs; do
    [ -d "$d" ] && rclone copy "$d" $R/bench/$NAME/$d --transfers 6 >/dev/null 2>&1
  done
  for d in bench_ipg20f60/*.h5dir bench_ipg8f60/*.h5dir; do
    [ -d "$d" ] && rclone copy "$d" $R/bench/$NAME/$d --transfers 6 >/dev/null 2>&1
  done
  for f in bench_*.log bench_*_time.log queue_*.txt hold12.txt bench_progress.log; do
    [ -f "$f" ] && rclone copy "$f" $R/bench/$NAME/logs/ >/dev/null 2>&1
  done
  echo "[$(date +%H:%M:%S)] $NAME 동기화"
  sleep $EVERY
done
