#!/bin/bash
# 절 CM117 묶음 extsolve 진입점 (exe/go_startpt.sh 꼴). 코드 디렉토리에서. 사용: bash exe/go_extsolve.sh <커밋> <단계[,단계]>
#   H9 wolf 의 선형 풀이 (확장 노드 소거 공간 Ã = Eᵀ A E 의 전처리) 판들. 판 명령 = R2 startpt/w9_1 (H9, 결과 불변 기록 감싸개 startpt f1~f50 + 렌더 덤프) 에서
#   감싸개 한 겹만 더하거나 (wolfh13 --w13, extsolve --xs) 예산·PCG 글자만 바꿈. H5 대조 = startpt/p5_1 (프레임 50, 기록 f1~f50, 덤프).
#   chk = 명령줄만 · r1 = 1 차 판 (b9 · m2 · m1 · e5 · pc · m2pc) · chk2·r2 = 체비셰프 평활 판 (x1·x2·x3 = --xs 1·2·3) · an = 집계 (CPU, an_runs.txt) · vid = 렌더 (vid_runs.txt)
# 판 디렉토리·덤프 = /home/dkta/work/extsolve/<판> (지우지 않음, R2 로 통째로 올림). 노드 / ·/tmp 안 씀: 코드·TMPDIR·캐시 /dev/shm/dkta.
export PATH=/home/dkta/bin:$PATH
H=$1; STG=$2
if [[ $STG == *,* ]]; then for s_ in ${STG//,/ }; do bash exe/go_extsolve.sh $H $s_ < /dev/null; done; exit 0; fi
R=r2:storage/result/anchorflow/lbours/wolfcv/extsolve
RS=r2:storage/result/anchorflow/lbours/wolfcv/startpt
O=/home/dkta/work/extsolve; C=$(pwd -P)
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_extsolve
PY=/home/dkta/.conda/envs/af/bin/python
L=$O/go_$STG.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_$STG.log -q; }
say "시작 $(hostname) 커밋 $H 단계 $STG; /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4, $5}') /home $(df -h /home | tail -1 | awk '{print $4, $5}') 남음; 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); 자료 $(ls /dev/shm/dkta/ag_data/wolf.pt /dev/shm/dkta/ag_data/gt_wolf/config.json 2>&1 | tr '\n' ' ')"
# 자료가 없으면 (새 잡) R2 에서 wolf.pt 를 받음; gt_wolf 는 R2 에 없어 없으면 멈춤 (기록만)
if [ ! -s /dev/shm/dkta/ag_data/wolf.pt ]; then mkdir -p /dev/shm/dkta/ag_data; rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wolf.pt /dev/shm/dkta/ag_data/wolf.pt -q; say "wolf.pt 받음 $(stat -c %s /dev/shm/dkta/ag_data/wolf.pt 2>&1)"; fi
[ -s /dev/shm/dkta/ag_data/gt_wolf/config.json ] || { say "gt_wolf 없음 -- 멈춤"; exit 1; }
base9() { rclone cat $RS/w9_1/cmd.txt | tr -d '\n' | sed -e 's/  */ /g' -e "s#--out [^ ]*\$#--out $O/$1#"; }
base5() { rclone cat $RS/p5_1/cmd.txt | tr -d '\n' | sed -e 's/  */ /g' -e "s#--out [^ ]*\$#--out $O/$1#" -e 's# --frames 12 # --frames 50 #' -e 's# --bar_cnt # --bar_cnt --dump --dump_part --dump_F #' -e 's#^ *#AF_SP_FR=1,50 #'; }
put() { mkdir -p $O/$1; echo "$2" > $O/$1/cmd.txt; rclone copyto $O/$1/cmd.txt $R/$1/cmd.txt -q; say "$1: $(grep -o -- 'wolfh13_wrap.py --w13 [0-9]\|extsolve_wrap.py[^.]*\.py\|--tb [0-9.]*\|--cg [0-9]*\|--tolP [0-9e.-]*\|--frames [0-9]*\|--out [^ ]*\|AF_SP_FR=[^ ]*' $O/$1/cmd.txt | tr '\n' ' ')"; }
w13() { sed "s# -u exe/startpt_wrap.py --sp 1 exe/wolfh12_wrap.py # -u exe/startpt_wrap.py --sp 1 exe/wolfh13_wrap.py --w13 $1 exe/wolfh12_wrap.py #"; }
xs() { sed "s# -u exe/startpt_wrap.py --sp 1 exe/wolfh12_wrap.py # -u exe/startpt_wrap.py --sp 1 exe/extsolve_wrap.py $1 exe/wolfh12_wrap.py #"; }
conv() { sed -e 's# --tb 16.7 # --tb 1000 #' -e 's# --cg 5 # --cg 40 #' -e 's# --tolP 1e-1 # --tolP 1e-2 #'; }
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
  env CUDA_VISIBLE_DEVICES=$g timeout 7200 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?; rm -f $CL/$g
  say "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) 칸 파일 $(ls $O/$n/sp 2>/dev/null | wc -l) 덤프 $(stat -c %s $O/$n/pred.pt 2>&1) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O/$n $R/$n -q --transfers 8; say "$n R2 올림 (덤프 포함, 노드 사본 남김)"; }
watch() { local END=$1; shift
  ( while [ ! -f $END ]; do sleep 120; for n in "$@"; do [ -f $O/$n/log.txt ] && { tail -c 3000 $O/$n/log.txt | rclone rcat $R/$n/log_tail.txt; }; done; done ) & }
runset() { local tag=$1; shift; rm -f $O/go_$tag.RUNEND; watch $O/go_$tag.RUNEND "$@"
  local PIDS="" n; for n in "$@"; do runone $n & PIDS="$PIDS $!"; sleep 20; done; wait $PIDS; touch $O/go_$tag.RUNEND; }
case $STG in
chk) B9=$(base9 b9); put b9 "$B9"
     put m2 "$(base9 m2 | w13 2)"; put m1 "$(base9 m1 | w13 1)"
     put e5 "$(base5 e5)"
     put pc "$(base9 pc | conv)"; put m2pc "$(base9 m2pc | w13 2 | conv)";;
chk2) for k in 1 2 3; do put x$k "$(base9 x$k | xs "--xs $k")"; done;;
r2)  for n in x1 x2 x3; do [ -s $O/$n/cmd.txt ] || { say "$n cmd.txt 없음 -- chk2 먼저"; exit 1; }; done
     runset r2 x1 x2 x3;;
ra)  for n in b9 m2 m1 e5 pc m2pc x1 x2 x3; do [ -s $O/$n/cmd.txt ] || { say "$n cmd.txt 없음 -- chk·chk2 먼저"; exit 1; }; done
     runset ra b9 m2 m1 e5 x1 x2 x3 pc m2pc;;
r1)  for n in b9 m2 m1 e5 pc m2pc; do [ -s $O/$n/cmd.txt ] || { say "$n cmd.txt 없음 -- chk 먼저"; exit 1; }; done
     runset r1 b9 m2 m1 e5 pc m2pc;;
an)  RUNS=$(cat $O/an_runs.txt 2>/dev/null || echo "b9 m2 m1 e5 x1 x2 x3 pc m2pc")
     $PY -u exe/extsolve_an.py $O $RUNS > $O/an.log 2>&1; say "an rc=$? $(tail -8 $O/an.log | tr '\n' ' ' | cut -c1-2500)"
     rclone copy $O $R --include "an.log" --include "extsolve_an.json" -q;;
vid) GTW=/dev/shm/dkta/ag_data/gt_wolf
     for n in $(cat $O/vid_runs.txt); do
       ( g=$(claim_wait $BASHPID); V=$O/$n/vj; HD=/dev/shm/dkta/extsolve_vid/$n; mkdir -p $V $HD
         $PY -u exe/lb_roll_to_h5.py --dump $O/$n/pred.pt --traj $GTW --out $HD/h5 > $V/h5.log 2>&1
         CUDA_VISIBLE_DEVICES=$g $PY -u exe/lb_render.py --sim $HD/h5 --config $GTW/config.json --shape wolf --out $V/fem_wolf.mp4 --frame_label 0 --caption " " < /dev/null > $V/r.log 2>&1
         rm -f $CL/$g; rm -rf $HD
         say "$n 영상 $(stat -c %s $V/fem_wolf.mp4 2>&1) $(tail -1 $V/r.log | cut -c1-160)"
         rclone copy $V $R/$n/vj -q ) & sleep 5
     done; wait;;
esac
echo "끝 $(date '+%F %T')" > $O/go_$STG.END; rclone copyto $O/go_$STG.END $R/go_$STG.END -q
say "끝"
