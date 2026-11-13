#!/bin/bash
# 절 CM133 묶음 (확장 T1E2 외부 노드 θ 섞기 extth, 예산 안 H9 + 수정, wolf 50 프레임). 코드 디렉토리에서. 사용: bash exe/go_cm133.sh <커밋> [카드0 판들] [카드1 판들] [렌더 판들]
#   원본 = R2 extsolve/nb9 (H9 예산) 명령 그대로, exe/extth_wrap.py <플래그> 를 ext2_wrap.py 앞에 끼우고 --out 만 바꿈 (go_cm132 와 같은 자리).
#   판: b9 = 원본 그대로 (같은 카드 프레임 시간 기준) · t1 = --xth 1 (θ = 점유 칸 안 기저 적분 몫) · t2 = --xth 2 (θ = m/(ρ̄ h³))
#       연기 (6 프레임): z1 = t1 · z2 = t2 -- 판 앞에 돌려 rc 0 · 유한 확인, 실패면 그 카드 줄 멈춤.
#   판 디렉토리 /home/dkta/work/cm133/<판> -> R2 lbours/wolfcv/cm133/<판> (덤프 포함, 안 지움). 끝에 an (convabl_an) · 렌더 (lb_roll_to_h5 · lb_render -> <판>/vj/fem_wolf.mp4).
export PATH=/home/dkta/bin:$PATH
H=$1; G0=${2:-z1,t1,b9}; G1=${3:-z2,t2}; VR=${4:-b9,t1,t2}
R=r2:storage/result/anchorflow/lbours/wolfcv/cm133
RX=r2:storage/result/anchorflow/lbours/wolfcv/extsolve
O=/home/dkta/work/cm133
PY=/home/dkta/.conda/envs/af/bin/python
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache
L=$O/go_run.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_run.log -q; }
flags() { case $1 in b9) echo "";; t1|z1) echo "--xth 1";; t2|z2) echo "--xth 2";; *) echo "?";; esac; }
mk() { local n=$1 F C; F=$(flags $n); [ "$F" = "?" ] && { say "$n 모르는 판"; return 1; }
  C=$(rclone cat $RX/nb9/cmd.txt | tr -d "\n")
  [ ${#C} -gt 100 ] || { say "nb9 명령 못 읽음"; return 1; }
  mkdir -p $O/$n
  if [ -n "$F" ]; then C=$(echo "$C" | sed -e "s#exe/speed12_wrap.py --s12z 1 exe/ext2_wrap.py#exe/speed12_wrap.py --s12z 1 exe/extth_wrap.py $F exe/ext2_wrap.py#"); fi
  case $n in z*) C=$(echo "$C" | sed -e 's#AF_SP_FR=1,50#AF_SP_FR=1,6#' -e 's# --frames 50 # --frames 6 #');; esac
  echo "$C" | sed -e "s#--out [^ ]*\$#--out $O/$n#" > $O/$n/cmd.txt
  rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q
  if [ -n "$F" ]; then grep -q "exe/extth_wrap.py $F exe/ext2_wrap.py --w8k 1 --w8e 1 --w8d 2 --w8t 1" $O/$n/cmd.txt || { say "$n 명령 고침 확인 실패"; return 1; }; fi
  case $n in z*) grep -q -- " --frames 6 " $O/$n/cmd.txt || { say "$n 프레임 고침 확인 실패"; return 1; };; *) grep -q -- " --frames 50 " $O/$n/cmd.txt || { say "$n 프레임 50 아님"; return 1; };; esac
  grep -q -- "--out $O/$n\$" $O/$n/cmd.txt || { say "$n --out 확인 실패"; return 1; }
  say "$n 명령: $(grep -o 'extth_wrap.py[^e]*exe/ext2_wrap.py\|--frames [0-9]*\|--tb [0-9.]*\|--newton [0-9]*\|--cg [0-9]*\|--out [^ ]*' $O/$n/cmd.txt | tr '\n' ' ')"; }
runone() { local n=$1 g=$2
  say "$n 카드 $g 시작 (계산 프로세스 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .))"
  ( while [ ! -f $O/$n/DONE ]; do sleep 90; tail -c 3000 $O/$n/log.txt 2>/dev/null | rclone rcat $R/$n/log_tail.txt; rclone copy $O/$n $R/$n -q --transfers 8 --exclude DONE; done ) &
  env CUDA_VISIBLE_DEVICES=$g timeout 5400 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?; touch $O/$n/DONE
  say "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(cat $O/$n/xtinfo.json 2>/dev/null | cut -c1-300) $(tail -2 $O/$n/xt.jsonl 2>/dev/null | tr '\n' ' ' | cut -c1-300) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O/$n $R/$n -q --transfers 8; say "$n R2 올림 (덤프 포함)"; return $rc; }
line() { local g=$1; shift; for n in "$@"; do runone $n $g || { case $n in z*) say "연기 $n 실패 -- 카드 $g 줄 멈춤"; return 1;; esac; }; done; }
for n in ${G0//,/ } ${G1//,/ }; do mk $n || exit 1; done
( line 0 ${G0//,/ } ) & P1=$!; sleep 20; ( line 1 ${G1//,/ } ) & P2=$!; wait $P1 $P2
RUNS=""; for n in ${G0//,/ } ${G1//,/ }; do case $n in z*) ;; *) [ -s $O/$n/summary.json ] && RUNS="$RUNS $n";; esac; done
$PY -u exe/convabl_an.py $O $RUNS > $O/an.log 2>&1; say "an rc=$? $(cat $O/an.log | tr '\n' ' ' | cut -c1-3000)"
rclone copy $O $R --include "an.log" --include "convabl_an.json" -q
GTW=/dev/shm/dkta/ag_data/gt_wolf; g=0
for n in ${VR//,/ }; do
  [ -s $O/$n/pred.pt ] || { say "$n pred.pt 없음 -- 렌더 안 함"; continue; }
  gg=$g; ( V=$O/$n/vj; HD=/dev/shm/dkta/cm133_vid/$n; mkdir -p $V $HD
    $PY -u exe/lb_roll_to_h5.py --dump $O/$n/pred.pt --traj $GTW --out $HD/h5 > $V/h5.log 2>&1
    CUDA_VISIBLE_DEVICES=$gg $PY -u exe/lb_render.py --sim $HD/h5 --config $GTW/config.json --shape wolf --out $V/fem_wolf.mp4 --frame_label 0 --caption " " < /dev/null > $V/r.log 2>&1
    rm -rf $HD
    say "$n 영상 $(stat -c %s $V/fem_wolf.mp4 2>&1) $(tail -1 $V/r.log | cut -c1-160)"
    rclone copy $V $R/$n/vj -q ) &
  g=$(( (g + 1) % 2 )); [ $g = 0 ] && wait
done; wait
say "끝 R2 대조 $(rclone check $O $R --one-way -q 2>&1 | grep -c ERROR) 개 차이"
