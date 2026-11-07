#!/bin/bash
# 절 CM23.3 묶음 wmk1conv: 수박 기본 판 K 1 (장벽 바닥 + 보간 교체, 뉴턴 2 x PCG 5, 허용치 εN 1e-2 · εP 1e-1) 의 첫 접촉 뒤 튐이 덜 풀림 탓인지.
#   같은 명령 (절 CM23 k1new M_K1 = exe/stab_k1new.sh base_of M + --K 1) 에서 반복 상한만: 2x5 (기본) · 4x10 · 8x20 · 16x40, 허용치 그대로.
#   진단 판 = + --nrec (결과 불변: 뉴턴 반복별 |g|/F, PCG, 보폭, 마지막 뉴턴 뒤 |g|/F, 질량중심 z) --qjd --jdiag --bar_cnt, 카드마다 하나씩 동시에.
#   단독 시간 판 = 진단 없음, 카드 하나에 차례로 (2x5 · 4x10 · 8x20 · 16x40).
#   연기: 결과 불변 확인 -- 6 프레임 --det 끔·켬 (--nrec) x_sha1 대조 (det 가 안 되면 비결정 판 오차 대조로 대신).
# 잡 안, 커밋 디렉토리에서; 출력 $1, R2 $2. 판이 끝나면 바로 R2.
O=$1; R=$2; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
WMD=/tmp/dkta/ag_data; mkdir -p $WMD
[ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
BASE="exe/fem_ao_rt_sim.py --data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh --fl_bar --fl_bst 1 --y0eb --lj_interp --fp_interp --tag wm --K 1"
DG="--nrec --qjd --jdiag --bar_cnt"
log() { echo "[$(date '+%F %T')] $*" >> $O/wmk1conv.log; rclone copyto $O/wmk1conv.log $R/wmk1conv.log -q; }
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
run1() {   # run1 이름 카드 추가인자...
  local n=$1 g=$2; shift 2; local c="$BASE $*"
  [ -f $O/$n/summary.json ] && return
  rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/gpu_before.txt 2>&1
  ( while sleep 10; do echo "$(date '+%T') 카드 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .) 노드 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" >> $O/$n/gpu_procs.txt; done ) & local wp=$!
  log "$n 카드 $g 시작"
  CUDA_VISIBLE_DEVICES=$g timeout 5400 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?; kill $wp 2>/dev/null
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-220) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n; }
# ---- 연기: 결과 불변 (6 프레임)
SM=$O/smoke; mkdir -p $SM
if [ ! -f $SM/ok ]; then
  g=$(claim_wait $$)
  for v in "det_off --det" "det_on --det $DG" "nd_off" "nd_on $DG"; do set -- $v; n=$1; shift
    mkdir -p $SM/$n; c="$(echo $BASE | sed 's/--frames 100/--frames 6/') --newton 2 --cg 5 $*"; echo "$PY -u $c --out $SM/$n" > $SM/$n/cmd.txt
    CUDA_VISIBLE_DEVICES=$g timeout 1800 $PY -u $c --out $SM/$n < /dev/null > $SM/$n/log.txt 2>&1; echo "rc=$?" > $SM/$n/rc.txt; done
  rm -f $CL/$g
  $PY - $SM > $SM/inv.txt 2>&1 <<'PYX'
import json, sys, os
d = sys.argv[1]
def ld(n):
    p = os.path.join(d, n, "summary.json")
    if not os.path.exists(p): return None, None
    s = json.load(open(p)); e = [json.loads(l)["err"] for l in open(os.path.join(d, n, "fem.jsonl"))]
    return s, e
a, ea = ld("det_off"); b, eb = ld("det_on"); c, ec = ld("nd_off"); dd, ed = ld("nd_on")
if a and b:
    print("DET", a["frames_done"], b["frames_done"], a["x_sha1"], b["x_sha1"], "SAME" if a["x_sha1"] == b["x_sha1"] and a["frames_done"] == b["frames_done"] == 6 else "DIFF")
else:
    print("DET 실패", bool(a), bool(b))
if c and dd:
    print("ND", c["frames_done"], dd["frames_done"], "err %", [round(100 * q, 4) for q in ec], [round(100 * q, 4) for q in ed])
PYX
  rclone copy $SM $R/smoke -q; log "[연기] $(cat $SM/inv.txt | tr '\n' ' ' | cut -c1-600)"
  if grep -q "^DET .*SAME" $SM/inv.txt; then touch $SM/ok
  elif grep -q "^DET .*DIFF" $SM/inv.txt; then echo "결과 불변 실패 $(date '+%F %T')" > $O/wmk1conv.FAIL; exit 1
  else
    $PY -c "
import json; e=[[json.loads(l)['err'] for l in open('$SM/'+n+'/fem.jsonl')] for n in ('nd_off','nd_on')]
import sys; sys.exit(0 if len(e[0])==len(e[1])==6 and all(abs(x-y)<5e-4 for x,y in zip(e[0][:3],e[1][:3])) else 1)" && touch $SM/ok || { echo "연기 실패 (det 안 됨, 비결정 대조 어긋남) $(date '+%F %T')" > $O/wmk1conv.FAIL; exit 1; }
  fi
fi
# ---- 진단 판 (카드마다 동시에)
for v in "N2P5 2 5" "N4P10 4 10" "N8P20 8 20" "N16P40 16 40"; do set -- $v
  ( me=$BASHPID; g=$(claim_wait $me); run1 d_$1 $g --newton $2 --cg $3 $DG; rm -f $CL/$g ) & sleep 3
done
wait
log "진단 판 끝"
$PY exe/wmk1conv_tab.py $O > $O/wmk1conv_table.md 2> $O/wmk1conv_table.err; rclone copy $O $R -q --include "wmk1conv_table.*"
# ---- 단독 시간 (카드 하나, 차례로, 진단 없음)
ZG=$(claim_wait $$); log "단독 시간 카드 $ZG"
for v in "N2P5 2 5" "N4P10 4 10" "N8P20 8 20" "N16P40 16 40"; do set -- $v; run1 z_$1 $ZG --newton $2 --cg $3; done
rm -f $CL/$ZG
$PY exe/wmk1conv_tab.py $O > $O/wmk1conv_table.md 2> $O/wmk1conv_table.err; rclone copy $O $R -q --include "wmk1conv_table.*"
log "끝"
