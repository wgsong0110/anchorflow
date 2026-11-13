#!/bin/bash
# 절 CM127 묶음 linwhy 진입점 (exe/go_lagpl.sh 꼴). 코드 디렉토리에서. 사용: bash exe/go_linwhy.sh <커밋> <단계[,단계]>
#   판 목록 = R2 lbours/wolfcv/linwhy/code/spec_<단계>.txt, 줄마다 '이름 기준판(R2 extsolve/<기준>/cmd.txt) 프레임수 감싸개인자(~ 는 빈칸) env(+ 로 이음, 예 LW_F=8,20+LW_IT=0) [바꿈 '--cg~100=>--cg~1000;...']'
#   단계 <이름> = 그 spec 의 판을 명령 만들고 (cmd.txt) 빈 카드에 동시에 돌림, an = 집계 (exe/linwhy_an.py).
# 판 디렉토리·덤프 = /home/dkta/work/linwhy/<판> (지우지 않음, R2 lbours/wolfcv/linwhy/<판> 로 통째로).
export PATH=/home/dkta/bin:$PATH
H=$1; STG=$2
if [[ $STG == *,* ]]; then for s_ in ${STG//,/ }; do bash exe/go_linwhy.sh $H $s_ < /dev/null; done; exit 0; fi
R=r2:storage/result/anchorflow/lbours/wolfcv/linwhy
RX=r2:storage/result/anchorflow/lbours/wolfcv/extsolve
O=/home/dkta/work/linwhy
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_linwhy
PY=/home/dkta/.conda/envs/af/bin/python
L=$O/go_$STG.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_$STG.log -q; }
say "시작 $(hostname) 커밋 $H 단계 $STG; 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
CL=/dev/shm/dkta/anerr_claim; mkdir -p $CL
NG=$(nvidia-smi -L | wc -l)
claim() { local me=$1 g f p
  for g in $(seq 0 $((NG - 1))); do
    f=$CL/$g
    if [ -f $f ]; then p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && continue; rm -f $f; fi
    [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" = 0 ] || continue
    ( set -o noclobber; echo $me > $f ) 2>/dev/null && { echo $g; return 0; }
  done; return 1; }
claim_wait() { local g; while :; do g=$(claim $1) && { echo $g; return; }; sleep 20; done; }
runone() { local n=$1 g; g=$(claim_wait $BASHPID); say "$n 카드 $g 시작"
  rm -rf $O/$n/sp $O/$n/lw $O/$n/lw.jsonl; mkdir -p $O/$n
  env CUDA_VISIBLE_DEVICES=$g timeout ${TMO:-9000} bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?; rm -f $CL/$g
  say "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) | $(grep -a 'CM127' $O/$n/log.txt | tail -3 | cut -c1-400 | tr '\n' ' ') lw 줄 $(cat $O/$n/lw.jsonl 2>/dev/null | wc -l) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O/$n $R/$n -q --transfers 8; say "$n R2 올림 (덤프 포함, 노드 사본 남김)"; }
watch() { local END=$1; shift
  ( while [ ! -f $END ]; do sleep 60; for n in "$@"; do [ -f $O/$n/log.txt ] && { tail -c 4000 $O/$n/log.txt | rclone rcat $R/$n/log_tail.txt; [ -f $O/$n/lw.jsonl ] && rclone copyto $O/$n/lw.jsonl $R/$n/lw.jsonl -q; [ -f $O/$n/lwinfo.json ] && rclone copyto $O/$n/lwinfo.json $R/$n/lwinfo.json -q; }; done; done ) & }
case $STG in
an)  RUNS=$(rclone cat $R/code/an_runs.txt 2>/dev/null)
     $PY -u exe/linwhy_an.py $O $RUNS > $O/an.log 2>&1; say "an rc=$? $(tail -30 $O/an.log | tr '\n' ' ' | cut -c1-3000)"
     rclone copyto $O/an.log $R/an.log -q;;
an2) RUNS=$(rclone cat $R/code/an2_runs.txt 2>/dev/null)
     $PY -u exe/extsolve_an.py $O $RUNS > $O/an2.log 2>&1; say "an2 rc=$? $(tail -12 $O/an2.log | tr '\n' ' ' | cut -c1-3000)"
     $PY -u exe/convabl_an.py $O $RUNS > $O/an2_cv.log 2>&1; say "an2_cv rc=$? $(tail -12 $O/an2_cv.log | tr '\n' ' ' | cut -c1-3000)"
     rclone copy $O $R --include "an2*.log" --include "extsolve_an.json" --include "convabl_an.json" -q;;
*)   rclone cat $R/code/spec_$STG.txt > $O/spec_$STG.txt || { say "spec_$STG 없음"; exit 1; }
     RUNS=""
     while read n base nf wargs envs reps; do [ -z "$n" ] && continue; [[ $n == \#* ]] && continue
       C=$(rclone cat $RX/$base/cmd.txt | tr -d "\n")
       [ ${#C} -gt 100 ] || { say "$base 명령 못 읽음"; continue; }
       wa=${wargs//\~/ }
       C=$(echo "$C" | sed -e "s#--out [^ ]*\$#--out $O/$n#" -e "s#--frames [0-9]*#--frames $nf#" -e "s# -u exe/startpt_wrap.py # -u exe/linwhy_wrap.py $wa exe/startpt_wrap.py #")
       if [ -n "$reps" ]; then IFS=';' read -ra RP <<< "$reps"; for rp in "${RP[@]}"; do fr=${rp%%=>*}; to=${rp#*=>}; C=$(echo "$C" | sed "s# ${fr//\~/ } # ${to//\~/ } #"); done; fi
       mkdir -p $O/$n; echo "${envs//+/ } $C" | sed 's/^ //' > $O/$n/cmd.txt
       rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q
       say "$n: $(cut -c1-160 $O/$n/cmd.txt) ... $(grep -o -- '--frames [0-9]*\|--newton [0-9]*\|--cg [0-9]*\|--out [^ ]*' $O/$n/cmd.txt | tr '\n' ' ')"
       RUNS="$RUNS $n"
     done < $O/spec_$STG.txt
     rm -f $O/go_$STG.RUNEND; watch $O/go_$STG.RUNEND $RUNS
     PIDS=""; for n in $RUNS; do runone $n & PIDS="$PIDS $!"; sleep 15; done; wait $PIDS; touch $O/go_$STG.RUNEND;;
esac
echo "끝 $(date '+%F %T')" > $O/go_$STG.END; rclone copyto $O/go_$STG.END $R/go_$STG.END -q
say "끝"
