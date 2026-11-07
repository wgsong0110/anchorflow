#!/bin/bash
# 절 CM36 묶음 speed2: 수박 실행기 커널 재작성 (--sp2, 반올림 수준) 확인·단독 시간, 셀 규칙 (가) 셀 3 (+ 조건부 3.5·4). 잡 안, 커밋 디렉토리에서.
# 사용: bash exe/stab_speed2.sh <출력> <R2> <단계>   -- 판 목록 = exe/speed2_spec.txt 의 '<단계> <이름> <씬> <방식> <플래그...>' 줄.
#   방식 n: 비결정 (카드 하나씩 동시, 수박 --erec) · q: 동기 벽시계 커널 프로파일 (exe/speed1_prof.py, 단독) · z: 단독 시간 a·b·c (비결정 --erec)
#   zc: z 와 같되 zM_on3_{a,b,c} 최악 프레임 최대 > 16.7 ms 일 때만 (조정자 결정 (2): 재작성 뒤에도 셀 3 이 넘으면 셀 3.5·4 판정)
#   단독 (q·z·zc) = 노드의 모든 카드에 다른 계산 프로세스 0 일 때만 시작 (exe/stab_speed1.sh 와 같음), 시작·끝 기록 (apps_*.txt).
O=$1; RO=$2; STG=$3; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
WMD=/tmp/dkta/ag_data; mkdir -p $WMD
[ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
A2K="--y0loc 3 --ue be --uf_res --ue_loc 2 --ls0_keep"                                       # 절 CM31 공통 후보 A2k
base_of() { case $1 in                                                                         # exe/stab_speed1.sh 글자 그대로
  M) echo "exe/fem_ao_rt_sim.py --data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --newton 2 --cg 5 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh --fl_bar --fl_bst 1 --y0eb --lj_interp --fp_interp --tag wm --K 1";;
esac; }
xa_of() { case $2 in q) echo "--prof 1 --erec";; *) echo "--erec";; esac; }
log() { echo "[$(date '+%F %T')] $*" >> $O/speed2_$STG.log; rclone copyto $O/speed2_$STG.log $RO/speed2_$STG.log -q; }
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
  c="$(base_of $s) $A2K $(xa_of $s $md) $x"; [ $md = q ] && c="exe/speed1_prof.py $c"
  rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
  case $md in q|z|zc) solo_wait;; esac
  { nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; cat /proc/loadavg; } > $O/$n/apps_at_start.txt 2>&1
  log "$n 카드 $g 시작 (노드 계산 프로세스 $(napps), load $(cut -d' ' -f1 /proc/loadavg))"
  CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  { nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; cat /proc/loadavg; } > $O/$n/apps_at_end.txt 2>&1
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n; }
SPEC=$(grep -v '^#' exe/speed2_spec.txt | awk -v s=$STG '$1==s')
# ---- 비결정 판 (카드마다 하나, 동시)
while read -r st n s md x; do
  [ -z "$n" ] && continue; [ "$md" = n ] || continue; [ -f $O/$n/summary.json ] && continue
  while [ $(jobs -rp | wc -l) -ge $NG ]; do sleep 5; done
  ( me=$BASHPID; g=$(claim_wait $me); run $n $s $md $g "$x"; rm -f $CL/$g ) & sleep 3
done <<< "$SPEC"
wait
log "동시 판 끝 $STG"
# ---- 위치 대조 (pred.pt 있는 판; c_off_a 기준)
if [ -s $O/c_off_a/pred.pt ] && [ ! -s $O/cmp_$STG.json ]; then
  $PY exe/speed2_tab.py cmp $O/c_off_a $O/c_off_b $(ls -d $O/c_on_* 2>/dev/null) > $O/cmp_$STG.json 2> $O/cmp_$STG.err; rclone copy $O $RO --include "cmp_$STG.*" -q
fi
# ---- 단독 (프로파일 q 한 번, 시간 z a·b·c, 조건부 zc a·b·c) -- 카드 하나, 차례로
if echo "$SPEC" | awk '$4=="q"||$4=="z"||$4=="zc"' | grep -q .; then
  ZG=$(claim_wait $$); log "단독 카드 $ZG"
  while read -r st n s md x; do [ "$md" = q ] && run $n $s $md $ZG "$x"; done <<< "$SPEC"
  for r in a b c; do
    while read -r st n s md x; do [ "$md" = z ] && run ${n}_$r $s z $ZG "$x"; done <<< "$SPEC"
  done
  W=$($PY exe/speed2_tab.py worst $O/zM_on3_a $O/zM_on3_b $O/zM_on3_c 2>/dev/null)
  log "셀 3 재작성 판 최악 프레임 최대 $W ms"
  if echo "$SPEC" | awk '$4=="zc"' | grep -q . && $PY -c "import sys; w=float(sys.argv[1]); sys.exit(0 if (w == w and w > 16.7) else 1)" "$W"; then
    log "16.7 ms 넘음 -> 조건부 판 (zc)"
    for r in a b c; do
      while read -r st n s md x; do [ "$md" = zc ] && run ${n}_$r $s zc $ZG "$x"; done <<< "$SPEC"
    done
  fi
  rm -f $CL/$ZG
fi
# ---- 표 재료
$PY exe/speed2_tab.py row $(ls -d $O/*/ | grep -v code) > $O/rows_$STG.jsonl 2> $O/rows_$STG.err
for d in $(ls -d $O/q*/ 2>/dev/null); do echo "$(basename $d) $($PY exe/speed2_tab.py prof $d)"; done > $O/prof_$STG.txt 2>&1
rclone copy $O $RO --include "rows_$STG.*" --include "prof_$STG.txt" -q
log "단계 끝 $STG"
