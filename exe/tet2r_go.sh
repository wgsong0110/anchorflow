#!/bin/bash
# 절 CM103 r 판 비교 영상 (사용자 11-10 요청): ficus·bread 의 r 판 (R2 lbours/tet2/rF_1·rB_1 cmd.txt) 과 H9 판 (rHF·rHB cmd.txt, '@t2 0' 기록만 = H9 결과 불변) 을
# 같은 글자에 --dump --dump_F --dump_part 만 붙여 한 판씩 다시 돌리고, 같은 카드에서 3DGS 렌더 (exe/stab_tet2.sh vid 와 같은 절차: lb_roll_to_h5 -> lb_render).
# 잡 133078 안 (exe/tet2r_boot.sh 가 부름; 코드 디렉토리에서). 덤프·h5 = /home/dkta/work/tet2r_vid (노드 / ·/tmp 안 씀), 카드 = 공용 표식 디렉토리로 계산 프로세스 0 인 카드만,
# R2 올리기는 판 끝난 뒤 따로 뒤에서 (실행을 막지 않음).
# 사용: bash exe/tet2r_go.sh <커밋> <판[,판]>   판: vrF vrB vhF vhB
export PATH=/home/dkta/bin:$PATH
H=$1; RUNS=$2
PY=/home/dkta/.conda/envs/af/bin/python
R2T=r2:storage/result/anchorflow/lbours/tet2
RO=r2:storage/result/anchorflow/lbours/tet2r
O=/home/dkta/work/tet2r_vid; mkdir -p $O
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache
L=$O/go.log
log() { echo "[$(date '+%F %T')] $*" >> $L; rclone copyto $L $RO/go.log -q; }
log "시작 $(hostname) 커밋 $H 판 $RUNS 코드 $(pwd -P); /home 여유 $(df -h --output=avail /home | tail -1); /dev/shm $(df --output=pcent /dev/shm | tail -1)"
source exe/fem_cm2_scn.sh
cm_data ficus; cm_data bread
[ -s /dev/shm/dkta/ag_data/bread.pt ] || { mkdir -p /dev/shm/dkta/ag_data; rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/bread.pt /dev/shm/dkta/ag_data/bread.pt.$$ -q && mv /dev/shm/dkta/ag_data/bread.pt.$$ /dev/shm/dkta/ag_data/bread.pt; }   # 판 글자가 이 경로를 씀
W=/home/dkta/work
FGT=$W/lbours/ficus_v0/gt
gtd() { local n=$1 d; for d in /dev/shm/dkta/ag_data/gt_$n /dev/shm/dkta/t2chain/gt_$n /dev/shm/dkta/rtrender1/gt_$n /dev/shm/dkta/hhtc/gt_$n /dev/shm/dkta/hhtc2/gt_$n /dev/shm/dkta/lightg3/gt_$n /dev/shm/dkta/gpw/gt_$n /dev/shm/dkta/gpw2/gt_$n /dev/shm/dkta/tet2/gt_$n; do [ -f $d/meta.json ] && [ -f $d/sim_0000000000.h5 ] && { echo $d; return; }; done; echo ""; }
GTB=$(gtd bread)
log "자료 ficus $(stat -c %s $FI 2>&1) bread $(stat -c %s /dev/shm/dkta/ag_data/bread.pt 2>&1) 정답 bread '$GTB' ficus $(ls $FGT/meta.json 2>&1)"
CL=/dev/shm/dkta/anerr_claim; mkdir -p $CL
NG=$(nvidia-smi -L | wc -l)
claim() { local me=$1 g f p
  for g in $(seq 0 $((NG - 1))); do
    f=$CL/$g
    if [ -f $f ]; then p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && continue; rm -f $f; fi
    [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" = 0 ] || continue
    ( set -o noclobber; echo $me > $f ) 2>/dev/null && { echo $g; return 0; }
  done; return 1; }
src_of() { case $1 in vrF) echo rF_1;; vrB) echo rB_1;; vhF) echo rHF;; vhB) echo rHB;; esac; }
one() {   # 판 하나: 카드 잡기 -> 계산 -> 렌더 -> 카드 놓기 -> 뒤에서 올리기
  local n=$1 src c g sc GTD CFG SH
  src=$(src_of $n); [ -z "$src" ] && { log "$n 모르는 판"; return; }
  case $n in *F) SH=ficus; GTD=$FGT; CFG=$FGT/config.json;; *B) SH=bread; GTD=$GTB; CFG=$GTB/config.json;; esac
  [ -z "$GTD" ] && { log "$n 정답 궤적 없음 -- 안 함"; return; }
  [ -s $O/$n/vj/fem_$SH.mp4 ] && { log "$n 영상 있음 -- 건너뜀"; return; }
  c=$(rclone cat $R2T/$src/cmd.txt) || { log "$n cmd.txt 못 받음"; return; }
  c=$(echo "$c" | sed -E "s#--out [^ ]+#--out $O/$n#")
  c="${c/ --tag $SH --K 1 / --tag $SH --K 1 --dump --dump_F --dump_part }"
  [[ "$c" == *" --dump_F "* ]] || { log "$n 플래그 못 붙임"; return; }
  case "$c" in *"/dev/shm/dkta/t2chain/gt_bread"*) c="${c//\/dev\/shm\/dkta\/t2chain\/gt_bread/$GTB}";; esac
  rm -rf $O/$n; mkdir -p $O/$n/vj; echo "$c" > $O/$n/cmd.txt
  while :; do g=$(claim $BASHPID) && break; sleep 30; done
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1
  log "$n ($src + 덤프) 카드 $g 시작"
  env CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_tet2r_$n CUDA_VISIBLE_DEVICES=$g timeout 7200 bash -c "$c" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  log "$n 계산 rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) pred $(stat -c %s $O/$n/pred.pt 2>&1) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  if [ -f $O/$n/pred.pt ]; then
    $PY -u exe/lb_roll_to_h5.py --dump $O/$n/pred.pt --traj $GTD --out $O/$n/h5 > $O/$n/vj/h5.log 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY -u exe/lb_render.py --sim $O/$n/h5 --config $CFG --shape $SH --out $O/$n/vj/fem_$SH.mp4 --frame_label 0 --caption " " < /dev/null > $O/$n/vj/r.log 2>&1
    rm -rf $O/$n/h5
    log "$n 영상 $(stat -c %s $O/$n/vj/fem_$SH.mp4 2>&1) $(tail -1 $O/$n/vj/r.log | cut -c1-160) $(tail -1 $O/$n/vj/h5.log | cut -c1-160)"
  fi
  [ "$(cat $CL/$g 2>/dev/null)" = "$BASHPID" ] && rm -f $CL/$g && log "$n 카드 $g 놓음"
  ( rclone copy $O/$n $RO/$n -q --exclude pred.pt; rclone copy $O/$n $RO/$n -q --include pred.pt; log "$n R2 올림 끝 (차이 $(rclone check $O/$n $RO/$n --one-way --size-only 2>&1 | grep -c ERROR))" ) > /dev/null 2>&1 < /dev/null & disown
}
for n in ${RUNS//,/ }; do one $n & sleep 20; done
wait
log "끝"
echo "끝 $(date '+%F %T')" > $O/go.END; rclone copyto $O/go.END $RO/go.END -q
