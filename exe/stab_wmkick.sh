#!/bin/bash
# 절 CM23.4 묶음 wmkick: 수박 기본 판 K 1 (장벽 바닥 + 보간 교체, 뉴턴 2 x PCG 5) 첫 접촉 f4 뒤 f5 부터 위로 날아가는 운동량의 출처 (장벽 / 탄성 되튐).
#   (1) 같은 명령 + --nrec --qjd --jdiag --bar_cnt --frec 15 (결과 불변): 뉴턴 반복별 노드 힘 몫 (탄성·장벽·관성·중력·잔차), 걸음 속도 변화 분해, 장벽 표본점, det F·탄성 에너지.
#       결과 불변 대조 = 같은 명령 진단 없음 (base0) 과 프레임별 오차 (비결정이라 해시 대신; 절 CM23.3 수준 차면 같음).
#   (2) 대조 (진단용 정의 변경, 기본 판은 그대로): 장벽 κ x0.1·x10, d̂ x2 (κ 정의대로 따름 / κ 고정), 예전 미끄럼 바닥 (--floor_slip, 장벽 없음); 8x20 은 wmk1conv d_N8P20 재사용.
#   모두 프레임 15, 비결정, 카드마다 하나씩 동시에 (공용 잡기 규약). 판이 끝나면 바로 R2.
O=$1; R=$2; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
WMD=/tmp/dkta/ag_data; mkdir -p $WMD
[ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
COM="exe/fem_ao_rt_sim.py --data $WMD/wm.pt --frames 15 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh --lj_interp --fp_interp --tag wm --K 1 --newton 2 --cg 5"
BAR="--fl_bar --fl_bst 1 --y0eb"
DG="--nrec --qjd --jdiag --frec 15"
log() { echo "[$(date '+%F %T')] $*" >> $O/wmkick.log; rclone copyto $O/wmkick.log $R/wmkick.log -q; }
up() { rclone copy $O/$1 $R/$1 -q --transfers 8; }
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
run1() {   # run1 이름 카드 인자...
  local n=$1 g=$2; shift 2; local c="$COM $*"
  [ -f $O/$n/summary.json ] && return
  rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/gpu_before.txt 2>&1
  log "$n 카드 $g 시작"
  CUDA_VISIBLE_DEVICES=$g timeout 1800 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-220) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n; }
# ---- 연기: 진단 판 3 프레임 (코드 오류 먼저)
if [ ! -f $O/smoke.ok ]; then
  g=$(claim_wait $$)
  mkdir -p $O/smoke; c="$(echo $COM | sed 's/--frames 15/--frames 5/') $BAR $DG --bar_cnt"
  CUDA_VISIBLE_DEVICES=$g timeout 1200 $PY -u $c --out $O/smoke < /dev/null > $O/smoke/log.txt 2>&1; echo "rc=$?" > $O/smoke/rc.txt
  rm -f $CL/$g; up smoke
  if grep -q '"fr": {"band_m"' $O/smoke/fem.jsonl 2>/dev/null && [ -f $O/smoke/summary.json ]; then touch $O/smoke.ok; log "[연기] 통과"
  else log "[연기] 실패 $(tail -5 $O/smoke/log.txt | tr '\n' ' ' | cut -c1-600)"; echo "연기 실패 $(date '+%F %T')" > $O/wmkick.FAIL; exit 1; fi
fi
# ---- 판 (카드마다 동시에)
V=("base $BAR $DG --bar_cnt" "base0 $BAR" "k01 $BAR $DG --bar_cnt --bar_kmul 0.1" "k10 $BAR $DG --bar_cnt --bar_kmul 10" "dh2 $BAR $DG --bar_cnt --bar_dhmul 2" "dh2k $BAR $DG --bar_cnt --bar_dhmul 2 --bar_dhk" "slip --floor_slip $DG")
for v in "${V[@]}"; do set -- $v; n=$1; shift; args="$*"
  ( me=$BASHPID; g=$(claim_wait $me); run1 $n $g $args; rm -f $CL/$g ) & sleep 3
done
wait
log "판 끝"
rclone copyto r2:storage/result/anchorflow/lbours/wmk1conv/d_N8P20/fem.jsonl $O/n8p20_fem.jsonl -q
$PY exe/wmkick_tab.py $O > $O/wmkick_table.md 2> $O/wmkick_table.err; rclone copy $O $R -q --include "wmkick_table.*"
log "끝"
