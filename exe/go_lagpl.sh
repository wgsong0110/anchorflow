#!/bin/bash
# 절 CM122 묶음 lagpl 진입점 (exe/go_extsolve.sh 꼴). 코드 디렉토리에서. 사용: bash exe/go_lagpl.sh <커밋> <단계[,단계]>
#   소성 지연 (exe/lagpl_wrap.py --xlag 1|2) 판. 명령 = R2 extsolve/ncvb (수렴 cvb + nlwhy 기록) · extsolve/nb9 (H9 예산 + 기록) 에서 감싸개 한 겹만 더함.
#   chk = 명령줄만 · run = c1·c2·b1·b2 (4 카드 동시) · an = 집계 · vid = 렌더 (R2 code/vid_runs.txt) · reg = 레고·ficus·bread (R2 code/reg_k.txt = xlag 모드)
# 판 디렉토리·덤프 = /home/dkta/work/lagpl/<판> (지우지 않음, R2 lbours/wolfcv/lagpl/<판> 로 통째로).
export PATH=/home/dkta/bin:$PATH
H=$1; STG=$2
if [[ $STG == *,* ]]; then for s_ in ${STG//,/ }; do bash exe/go_lagpl.sh $H $s_ < /dev/null; done; exit 0; fi
R=r2:storage/result/anchorflow/lbours/wolfcv/lagpl
RX=r2:storage/result/anchorflow/lbours/wolfcv/extsolve
O=/home/dkta/work/lagpl
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_lagpl
PY=/home/dkta/.conda/envs/af/bin/python
L=$O/go_$STG.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_$STG.log -q; }
say "시작 $(hostname) 커밋 $H 단계 $STG; 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); 자료 $(ls /dev/shm/dkta/ag_data/wolf.pt /dev/shm/dkta/ag_data/gt_wolf/config.json 2>&1 | tr '\n' ' ')"
put() { mkdir -p $O/$1; echo "$2" > $O/$1/cmd.txt; rclone copyto $O/$1/cmd.txt $R/$1/cmd.txt -q; say "$1: $(grep -o -- 'lagpl_wrap.py --xlag [0-9]\|--newton [0-9]*\|--tb [0-9.]*\|--cg [0-9]*\|--tolN [^ ]*\|--frames [0-9]*\|--out [^ ]*' $O/$1/cmd.txt | tr '\n' ' ')"; }
ins() { sed "s# exe/wolfh12_wrap.py # exe/lagpl_wrap.py $1 exe/wolfh12_wrap.py #"; }
o() { sed "s#--out [^ ]*\$#--out $O/$1#"; }
CL=/dev/shm/dkta/anerr_claim; mkdir -p $CL
NG=$(nvidia-smi -L | wc -l)
claim() { local me=$1 g f p
  for g in $(seq 0 $((NG - 1))); do
    f=$CL/$g
    if [ -f $f ]; then p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && continue; rm -f $f; fi
    [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" = 0 ] || continue
    ( set -o noclobber; echo $me > $f ) 2>/dev/null && { echo $g; return 0; }
  done; return 1; }
claim_wait() { local g; while :; do g=$(claim $1) && { echo $g; return; }; sleep 30; done; }
runone() { local n=$1 g; g=$(claim_wait $BASHPID); say "$n 카드 $g 시작"
  rm -rf $O/$n/sp; mkdir -p $O/$n
  env CUDA_VISIBLE_DEVICES=$g timeout 9000 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?; rm -f $CL/$g
  say "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(grep -a 'CM122' $O/$n/log.txt | tail -1 | cut -c1-160) 칸 파일 $(ls $O/$n/sp 2>/dev/null | wc -l) 덤프 $(stat -c %s $O/$n/pred.pt 2>&1) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O/$n $R/$n -q --transfers 8; say "$n R2 올림 (덤프 포함, 노드 사본 남김)"; }
watch() { local END=$1; shift
  ( while [ ! -f $END ]; do sleep 120; for n in "$@"; do [ -f $O/$n/log.txt ] && { tail -c 3000 $O/$n/log.txt | rclone rcat $R/$n/log_tail.txt; [ -f $O/$n/nl.jsonl ] && rclone copyto $O/$n/nl.jsonl $R/$n/nl.jsonl -q; }; done; done ) & }
runset() { local tag=$1; shift; rm -f $O/go_$tag.RUNEND; watch $O/go_$tag.RUNEND "$@"
  local PIDS="" n; for n in "$@"; do runone $n & PIDS="$PIDS $!"; sleep 20; done; wait $PIDS; touch $O/go_$tag.RUNEND; }
case $STG in
chk) C=$(rclone cat $RX/ncvb/cmd.txt | tr -d "\n"); B=$(rclone cat $RX/nb9/cmd.txt | tr -d "\n")
     [ ${#C} -gt 100 ] && [ ${#B} -gt 100 ] || { say "ncvb·nb9 명령 못 읽음"; exit 1; }
     put c1 "$(echo "$C" | o c1 | ins '--xlag 1')"
     put c2 "$(echo "$C" | o c2 | ins '--xlag 2')"
     put b1 "$(echo "$B" | o b1 | ins '--xlag 1')"
     put b2 "$(echo "$B" | o b2 | ins '--xlag 2')";;
run) for n in c1 c2 b1 b2; do [ -s $O/$n/cmd.txt ] || { say "$n cmd.txt 없음 -- chk 먼저"; exit 1; }; done
     runset run c1 c2 b1 b2;;
chk2) # 추가 판 (R2 code/extra_runs.txt 줄마다 '이름 기준(ncvb|nb9) 감싸개인자...') -- 1 차 결과를 본 뒤 정함
     rclone cat $R/code/extra_runs.txt | while read n base args; do [ -z "$n" ] && continue
       C=$(rclone cat $RX/$base/cmd.txt | tr -d "\n"); put $n "$(echo "$C" | o $n | ins "$args")"; done;;
run2) RUNS=$(rclone cat $R/code/extra_runs.txt | awk '{print $1}' | tr '\n' ' ')
     for n in $RUNS; do [ -s $O/$n/cmd.txt ] || { say "$n cmd.txt 없음 -- chk2 먼저"; exit 1; }; done
     runset run2 $RUNS;;
an)  RUNS=$(rclone cat $R/code/an_runs.txt 2>/dev/null || echo "c1 c2 b1 b2")
     $PY -u exe/extsolve_an.py $O $RUNS > $O/an.log 2>&1; say "an rc=$? $(tail -8 $O/an.log | tr '\n' ' ' | cut -c1-2500)"
     $PY -u exe/convabl_an.py $O $RUNS > $O/an_cv.log 2>&1; say "an_cv rc=$? $(tail -8 $O/an_cv.log | tr '\n' ' ' | cut -c1-2500)"
     rclone copy $O $R --include "an.log" --include "an_cv.log" --include "extsolve_an.json" --include "convabl_an.json" -q;;
vid) GTW=/dev/shm/dkta/ag_data/gt_wolf
     for n in $(rclone cat $R/code/vid_runs.txt); do
       ( g=$(claim_wait $BASHPID); V=$O/$n/vj; HD=/dev/shm/dkta/lagpl_vid/$n; mkdir -p $V $HD
         $PY -u exe/lb_roll_to_h5.py --dump $O/$n/pred.pt --traj $GTW --out $HD/h5 > $V/h5.log 2>&1
         CUDA_VISIBLE_DEVICES=$g $PY -u exe/lb_render.py --sim $HD/h5 --config $GTW/config.json --shape wolf --out $V/fem_wolf.mp4 --frame_label 0 --caption " " < /dev/null > $V/r.log 2>&1
         rm -f $CL/$g; rm -rf $HD
         say "$n 영상 $(stat -c %s $V/fem_wolf.mp4 2>&1) $(tail -1 $V/r.log | cut -c1-160)"
         rclone copy $V $R/$n/vj -q ) & sleep 5
     done; wait;;
reg) # 다른 씬 회귀: speed13 oXL_1·oXF_1·oXB_1 명령 그대로와 --xlag K (모래 아닌 씬은 ao_elem_s 를 안 써서 1 은 같은 결과여야 함)
     K=$(rclone cat $R/code/reg_k.txt)
     for sc in oXL_1 oXF_1 oXB_1; do
       c=$(rclone cat r2:storage/result/anchorflow/lbours/speed13/$sc/cmd.txt | tr -d '\n' | sed -e 's/  */ /g')
       miss=""; for f in $(echo "$c" | grep -o -- '--data [^ ]*\|--rr_traj [^ ]*' | awk '{print $2}'); do [ -e $f ] || miss="$miss $f"; done
       [ -n "$miss" ] && { say "$sc 자료 없음:$miss -- 건너뜀"; continue; }
       for k in 0 $K; do n=g${sc:2:1}_$k
         cc=$(echo "$c" | sed "s#--out [^ ]*\$#--out $O/$n#")
         [ "$k" != 0 ] && cc=$(echo "$cc" | sed "s# -u exe/speed13_wrap.py # -u exe/lagpl_wrap.py --xlag $k exe/speed13_wrap.py #")
         put $n "$cc"; runone $n & sleep 20; done; done; wait;;
esac
echo "끝 $(date '+%F %T')" > $O/go_$STG.END; rclone copyto $O/go_$STG.END $R/go_$STG.END -q
say "끝"
