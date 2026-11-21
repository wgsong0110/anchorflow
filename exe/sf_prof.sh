#!/bin/bash
# [CM192 측정] 우리 판 사다리 명령 한 판 (기본 ship N1, 12 프레임) 의 GPU 사용률이 낮은 원인 실측 (인스턴스 안, 코드 폴더에서).
#   A: 그대로 + pidstat -t (스레드별 CPU) + nvidia-smi pmon (이 프로세스 SM%) + 프레임 시간
#   B: python -m cProfile 로 같은 명령 (파이썬 함수별 시간; GPU 동기 지점 (.item·.cpu·synchronize) 에서 기다린 시간이 그 함수에 잡힘)
#   결과 -> R2 lbours/stabfinal/diag192/prof/<태그>/
#   bash exe/sf_prof.sh <장면> <N> <프레임> <태그> <드라이버 추가 인자...>
s=$1; n=$2; fr=$3; tag=$4; shift 4
O=/root/sf/prof_$tag; mkdir -p $O; RX=r2:storage/result/anchorflow/lbours/stabfinal
command -v pidstat >/dev/null || (apt-get install -y -qq sysstat >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq sysstat) >/dev/null 2>&1)
CMD=$(python -u exe/stab1115_drv.py --scene $s --stages cmd --gt_ns $n --gt_fr $fr "$@" 2>/dev/null | grep '^CMD ' | head -1 | cut -c5-)
echo "$CMD" > $O/cmd.txt
D=/home/dkta/work/stab/$s/prof/N$n
run() {   # $1 = 이름, $2 = 명령
  rm -rf $D; mkdir -p $D
  bash -c "$2" > $O/$1.log 2>&1 & P=$!
  PY=""; for k in $(seq 1 30); do sleep 1; PY=$(ps -eo pid,args | grep "sh_npz $D/res.npz" | grep -v -E "grep|bash -c" | awk '{print $1}' | tail -1); [ -n "$PY" ] && break; done
  echo "pid $PY" > $O/$1.pid
  pidstat -t -u -p $PY 2 > $O/$1.pidstat 2>&1 & S1=$!
  nvidia-smi pmon -d 2 > $O/$1.pmon 2>&1 & S2=$!
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -lms 2000 > $O/$1.util 2>&1 & S3=$!
  t0=$(date +%s.%N); wait $P; t1=$(date +%s.%N)
  kill $S1 $S2 $S3 2>/dev/null
  echo "벽시계 $(python -c "print(round($t1-$t0,1))") s" > $O/$1.wall
  cp $D/res.npz $O/$1_res.npz 2>/dev/null
}
run A "$CMD"
PYB=$(echo "$CMD" | sed -E 's#(python[^ ]* -u )#\1-m cProfile -o '"$O"'/B.prof #')
echo "$PYB" > $O/cmdB.txt
run B "$PYB"
python - <<PY > $O/B_top.txt 2>&1
import pstats
p = pstats.Stats("$O/B.prof"); p.sort_stats("tottime").print_stats(40); p.sort_stats("cumulative").print_stats(40)
PY
for r in A B; do P2=$(cut -d' ' -f2 $O/$r.pid); awk -v p="$P2" '$2==p && $4!="-" {s+=$4; n++} END{printf "이 프로세스 SM %% 평균 %.1f (표본 %d)\n", (n?s/n:0), n}' $O/$r.pmon > $O/$r.sm; done
rclone copy $O $RX/diag192/prof/$tag -q
echo "[sf_prof] 끝 $tag"
