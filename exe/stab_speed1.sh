#!/bin/bash
# 절 CM34 묶음 speed1: 공통 후보 A2k 의 wolf·수박 최악 프레임 프로파일·결과 불변 가속·수박 셀 크기. 잡 안, 커밋 디렉토리에서.
# 사용: bash exe/stab_speed1.sh <출력> <R2> <단계>   -- 판 목록 = exe/speed1_spec.txt 의 '<단계> <이름> <씬> <방식> <플래그...>' 줄.
#   방식 d: 결정적 + 해시 (카드 하나씩 동시) · n: 비결정 (카드 하나씩 동시, 수박 --erec) · p: 프로파일 (비결정 + --prof 1, 단독) · z: 단독 시간 a·b (비결정, 수박 --erec)
#   단독 (p·z) = 노드의 모든 카드에 다른 계산 프로세스 0 일 때만 시작 (아니면 60 초마다 다시 봄), 끝날 때 다시 셈 (apps_*.txt·load).
O=$1; RO=$2; STG=$3; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
source exe/fem_cm2_scn.sh
for t in L000 ficus bread wolf; do cm_data $t; done
WMD=/tmp/dkta/ag_data; mkdir -p $WMD
[ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
BA0="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2 --ga_y0ut --hspec --ncomp --ga hht --rho 0.8"
FBB="--floor vel --fl_bar --fl_bst 1 --y0eb"
TL="--tolN 1e-2 --tolP 1e-1"
A2K="--y0loc 3 --ue be --uf_res --ue_loc 2 --ls0_keep"                                       # 절 CM31 공통 후보 A2k
base_of() { case $1 in                                                                         # exe/stab_common1.sh 글자 그대로
  L) echo "exe/fem_an_rt_sim.py --data $SD/L000.pt --frames 80 $FBB $BA0 --newton 2 --cg 4 $TL --fp_interp --tag L000 --K 1";;
  F) echo "exe/fem_an_rt_sim.py --data $FI --frames 125 --floor none --cfg_bc --region_density --pg_vdamp $BA0 --newton 2 --cg 24 $TL --fp_interp --tag ficus --K 1";;
  B) echo "exe/fem_an_rt_sim.py --data $AG/bread.pt --frames 300 --floor none --pbc --pg_vdamp --visc_rpic --mlow 1e-3 $BA0 --newton 2 --cg 2 $TL --fp_interp --tag bread --K 1";;
  W) echo "exe/fem_an_rt_sim.py --data $AG/wolf.pt --frames 50 $FBB --pbc --mlow 1e-3 $BA0 --cell 2 --newton 2 --cg 5 $TL --fp_interp --tag wolf --K 1";;
  M) echo "exe/fem_ao_rt_sim.py --data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --newton 2 --cg 5 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh --fl_bar --fl_bst 1 --y0eb --lj_interp --fp_interp --tag wm --K 1";;
esac; }
xa_of() { local s=$1 md=$2
  case $md in
    d) case $s in W) echo "--det --hgt --fhash --bar_cnt";; L) echo "--det --hgt --fhash";; F|B) echo "--det --fhash";; M) echo "--erec";; esac;;   # 수박은 플래그로 --det
    q) echo "--prof 1$([ $s = M ] && echo ' --erec')";;
    n|z) [ $s = M ] && echo "--erec";;
    p) echo "--prof 1$([ $s = M ] && echo ' --erec')";;
  esac; }
log() { echo "[$(date '+%F %T')] $*" >> $O/speed1_$STG.log; rclone copyto $O/speed1_$STG.log $RO/speed1_$STG.log -q; }
up() { rclone copy $O/$1 $RO/$1 -q --transfers 8 --s3-upload-concurrency 16 --s3-chunk-size 64M; }
CL=/dev/shm/dkta/anerr_claim; mkdir -p $CL
NG=$(nvidia-smi -L | wc -l)
napps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .; }
claim() { local me=$1 g f p
  for g in $(seq 0 $((NG - 1))); do
    f=$CL/$g
    if [ -f $f ]; then p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && continue; rm -f $f; fi
    [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" = 0 ] || continue
    ( set -o noclobber; echo $me > $f ) 2>/dev/null && { echo $g; return 0; }
  done; return 1; }
claim_wait() { local g; while :; do g=$(claim $1) && { echo $g; return; }; sleep 30; done; }
solo_wait() { local w=0; while [ "$(napps)" != 0 ]; do [ $((w % 10)) = 0 ] && log "단독 기다림: 노드 계산 프로세스 $(napps) ($(cat /proc/loadavg | cut -d' ' -f1-3))"; w=$((w + 1)); sleep 60; done; }
run() {   # run 이름 씬 방식 카드 플래그
  local n=$1 s=$2 md=$3 g=$4 x=$5 c
  [ -f $O/$n/summary.json ] && return
  c="$(base_of $s) $A2K $(xa_of $s $md) $x"; [ $md = q ] && c="exe/speed1_prof.py $c"            # q: 동기 벽시계 커널 프로파일 (exe/speed1_prof.py)
  rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
  case $md in p|q|z) solo_wait;; esac
  { nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; cat /proc/loadavg; } > $O/$n/apps_at_start.txt 2>&1
  log "$n 카드 $g 시작 (노드 계산 프로세스 $(napps), load $(cut -d' ' -f1 /proc/loadavg))"
  CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  { nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; cat /proc/loadavg; } > $O/$n/apps_at_end.txt 2>&1
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n; }
SPEC=$(grep -v '^#' exe/speed1_spec.txt | awk -v s=$STG '$1==s')
# ---- 결정적·비결정 판 (카드마다 하나, 동시)
while read -r st n s md x; do
  [ -z "$n" ] && continue; case $md in d|n) ;; *) continue;; esac; [ -f $O/$n/summary.json ] && continue
  while [ $(jobs -rp | wc -l) -ge $NG ]; do sleep 5; done
  ( me=$BASHPID; g=$(claim_wait $me); run $n $s $md $g "$x"; rm -f $CL/$g ) & sleep 3
done <<< "$SPEC"
wait
log "동시 판 끝 $STG"
# ---- 단독 (프로파일 p 한 번, 시간 z a·b) -- 카드 하나, 차례로, 노드에 다른 계산 프로세스 없을 때
if echo "$SPEC" | awk '$4=="p"||$4=="q"||$4=="z"' | grep -q .; then
  ZG=$(claim_wait $$); log "단독 카드 $ZG"
  while read -r st n s md x; do case "$md" in p|q) run $n $s $md $ZG "$x";; esac; done <<< "$SPEC"
  for r in a b; do
    while read -r st n s md x; do [ "$md" = z ] && run ${n}_$r $s z $ZG "$x"; done <<< "$SPEC"
  done
  rm -f $CL/$ZG
fi
# ---- 상자 진단 (읽기만): 정답·덤프한 wolf 예측의 입자 상자
PW=$(ls $O/W_dump*/pred.pt 2>/dev/null | tr '\n' ' ')
if [ -n "$PW" ] && [ ! -s $O/bbox_$STG.txt ]; then $PY exe/speed1_bbox.py $AG/wolf.pt 0.02 $PW > $O/bbox_$STG.txt 2>&1; rclone copyto $O/bbox_$STG.txt $RO/bbox_$STG.txt -q; fi
log "단계 끝 $STG"
