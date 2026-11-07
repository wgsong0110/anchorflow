#!/bin/bash
# 절 CM23 ficus 영상 (사용자 요청 11-07): 정답 | 기본 판 K 1 | K 2 비교, 3DGS 알파블렌딩 (절 CM4 v_ficus 와 같은 렌더: lb_roll_to_h5 -> lb_render --shape ficus).
#   k1new F_K1·F_K2 덤프 (pred.pt) 에는 F 가 없어 (--dump_F 없음) 렌더 불가 -> 같은 명령 (R2 cmd.txt 그대로, 결정적 --det) 에 --dump_F 만 붙여 다시.
#   결과 불변 확인: 끝 해시 x_sha1 이 k1new 판과 같아야 함 (다르면 렌더 안 함).
# 잡 안, 커밋 디렉토리에서; 출력 $1, R2 $2. 판마다 빈 카드 하나 (시뮬 -> h5 -> 렌더), 동시에.
O=$1; R=$2; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
source exe/fem_cm2_scn.sh; cm_data ficus
W=/home/dkta/work; GTD=$W/lbours/ficus_v0/gt
log() { echo "[$(date '+%F %T')] $*" >> $O/fk1vid.log; rclone copyto $O/fk1vid.log $R/fk1vid.log -q; }
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
log "카드 $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '); 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
one() {   # one F_K1|F_K2
  local n=$1 me=$BASHPID g
  g=$(claim_wait $me); log "$n 카드 $g 시작"
  mkdir -p $O/$n/ref; rclone copy r2:storage/result/anchorflow/lbours/k1new/$n $O/$n/ref -q --include cmd.txt --include summary.json --include period.txt
  local c; c=$(sed -e "s|--out [^ ]*|--out $O/$n|" -e "s|^[^ ]*python|$PY|" $O/$n/ref/cmd.txt)" --dump_F"
  echo "$c" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1
  [ -f $O/$n/summary.json ] || CUDA_VISIBLE_DEVICES=$g timeout 3600 $c < /dev/null > $O/$n/log.txt 2>&1
  local h0 h1
  h0=$($PY -c "import json; print(json.load(open('$O/$n/ref/summary.json'))['x_sha1'])")
  h1=$($PY -c "import json; print(json.load(open('$O/$n/summary.json'))['x_sha1'])" 2>/dev/null)
  log "$n 시뮬 끝 해시 $h1 (k1new $h0) $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  if [ "$h0" != "$h1" ]; then log "$n 해시 다름 -> 렌더 안 함"; rm -f $CL/$g; rclone copy $O/$n $R/$n -q --exclude pred.pt; return; fi
  $PY exe/fem_p_period.py --data $FI --pred $O/$n/pred.pt > $O/$n/period.txt 2>&1
  [ -f $O/$n/h5/meta.json ] || $PY -u exe/lb_roll_to_h5.py --dump $O/$n/pred.pt --traj $GTD --out $O/$n/h5 > $O/$n/h5.log 2>&1
  mkdir -p $O/$n/vj
  CUDA_VISIBLE_DEVICES=$g $PY -u exe/lb_render.py --sim $O/$n/h5 --config $GTD/config.json --shape ficus --out $O/$n/vj/fem_ficus.mp4 --frame_label 0 --caption " " < /dev/null > $O/$n/vj/r.log 2>&1
  log "$n 렌더 $(tail -1 $O/$n/h5.log | cut -c1-150) $(ls -la $O/$n/vj/fem_ficus.mp4 2>&1 | awk '{print $5}') $(tail -1 $O/$n/vj/r.log | cut -c1-150) | $(tail -1 $O/$n/period.txt)"
  rm -f $CL/$g
  rclone copy $O/$n $R/$n -q --exclude pred.pt --exclude "h5/**"; }
for n in F_K1 F_K2; do one $n & sleep 3; done
wait
log "끝"
