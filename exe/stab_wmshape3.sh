#!/bin/bash
# 절 CM44 묶음 wmshape3: 수박 깨짐 모양 3 -- 늦은 손상 기어오름 막기 후보 (--ws3, lib/anchorflow/fem_wmshape3.py) 를 BW4 위에서. 잡 안, 커밋 디렉토리에서.
# 사용: bash exe/stab_wmshape3.sh <출력> <R2> <단계>   -- 판 목록 = exe/wmshape3_spec.txt 의 '<단계> <이름> <씬> <방식> <플래그...>' 줄.
#   씬 M3 = 수박 셀 3 옛 실행기 (절 CM40 stab_wmshape.sh 그대로). 방식 r = --erec + 진단 (--wsh_rec·--ws2), z = 단독 시간 (--erec 만, 카드 하나에 차례로 a·b).
#   정답 통계 = 절 CM40 gt_wmshape.json · 절 CM41 gt_wmshape2.json 복사.
O=$1; RO=$2; STG=$3; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
WMD=/tmp/dkta/ag_data; mkdir -p $WMD
[ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
JP=/dev/shm/dkta/ag_data/gt_jp; mkdir -p $JP
[ $(ls $JP | wc -l) -ge 101 ] || rclone copy r2:storage/result/anchorflow/lbours/fem_cm/wmJd/gt_jp $JP -q --transfers 16
GH5=/home/dkta/work/wm2/gf/simulation_ply
WMB="exe/fem_ao_rt_sim.py --data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --newton 2 --cg 5 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh --fl_bar --fl_bst 1 --y0eb --lj_interp --fp_interp --tag wm --K 1 --gq_oct 0 --ws_bar 4 --ws_fi"
base_of() { case $1 in M3) echo "${WMB/--cell 1.5/--cell 3}";; esac; }
nw() { case " $1 " in *" --newton "*) echo "$1" | sed 's/.*--newton \([0-9]*\).*/\1/';; *) echo "";; esac; }
SVF=3,4,5,6,8,10,15,20,30,50,100
dg_of() { case $1 in r) echo "--erec --wsh_rec 30 --wsh_sv $SVF --ws2 1";; z|b) echo "--erec";; e) echo "--erec --wsh_rec 30 --nrec --enrec 12";; esac; }
log() { echo "[$(date '+%F %T')] $*" >> $O/wmshape3_$STG.log; rclone copyto $O/wmshape3_$STG.log $RO/wmshape3_$STG.log -q; }
up() { rclone copy $O/$1 $RO/$1 -q --transfers 8 --exclude pred.pt; }
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
run() {   # run 이름 씬 방식 카드 플래그
  local n=$1 s=$2 md=$3 g=$4 x=$5 c b
  [ -f $O/$n/summary.json ] && return
  b="$(base_of $s)"; [ -n "$(nw "$x")" ] && b="${b/--newton 2/--newton $(nw "$x")}" && x="$(echo "$x" | sed 's/--newton [0-9]*//')"
  c="$b $(dg_of $md) $x"
  rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1
  log "$n 카드 $g 시작"
  CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_end.txt 2>&1
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n; }
SPEC=$(grep -v '^#' exe/wmshape3_spec.txt | awk -v s=$STG '$1==s')
mkdir -p $O/gt; [ -s $O/gt/gt_wmshape.json ] || rclone copyto r2:storage/result/anchorflow/lbours/wmshape/gt/gt_wmshape.json $O/gt/gt_wmshape.json -q   # 절 CM40 정답 통계 그대로
[ -s $O/gt/gt_wmshape2.json ] || rclone copyto r2:storage/result/anchorflow/lbours/wmshape2/gt/gt_wmshape2.json $O/gt/gt_wmshape2.json -q   # 절 CM41 정답 손상 입자 상태 그대로
while read -r st n s md x; do
  [ -z "$n" ] && continue; [ $md = z ] && continue; [ -f $O/$n/summary.json ] && continue
  while [ $(jobs -rp | wc -l) -ge $NG ]; do sleep 5; done
  ( me=$BASHPID; g=$(claim_wait $me); run $n $s $md $g "$x"; rm -f $CL/$g ) & sleep 3
done <<< "$SPEC"
wait
log "판정 판 끝 $STG"
if echo "$SPEC" | awk '$4=="z"' | grep -q .; then
  ZG=$(claim_wait $$); log "단독 시간 카드 $ZG"
  W0=$(date +%s); while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" != 0 ] && [ $(( $(date +%s) - W0 )) -lt 1200 ]; do sleep 20; done   # 단독: 노드 계산 프로세스 0 까지 (최대 20 분)
  log "단독 시간 앞 기다림 $(( $(date +%s) - W0 )) 초, 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)"
  for r in a b; do
    while read -r st n s md x; do
      [ "$md" = z ] || continue; run ${n}_$r $s z $ZG "$x"
    done <<< "$SPEC"
  done
  rm -f $CL/$ZG
fi
if [ $STG = s0 ]; then                                                                      # 연기: 모든 판이 5 프레임 넘게 끝나야
  OK=1; while read -r st n s md x; do [ -z "$n" ] && continue
    $PY -c "import json,sys; s=json.load(open('$O/$n/summary.json')); sys.exit(0 if s['frames_done'] >= 5 and not s.get('diverged') else 1)" 2>/dev/null || OK=0
    grep -aq '"wsh": {' $O/$n/fem.jsonl 2>/dev/null || OK=0
    grep -aq '"br": \[' $O/$n/fem.jsonl 2>/dev/null || OK=0
    grep -aq '"ws3": {' $O/$n/fem.jsonl 2>/dev/null || OK=0
  done <<< "$SPEC"
  [ -s $O/gt/gt_wmshape2.json ] || OK=0
  log "[연기] OK=$OK"; [ $OK = 1 ] || { echo "연기 실패 $(date '+%F %T')" > $O/wmshape3.FAIL; rclone copy $O $RO -q --include wmshape3.FAIL; exit 1; }
fi
[ $STG = s0 ] || { $PY exe/wmshape3_tab.py $O $JP > $O/wmshape3_table.md 2> $O/wmshape3_table.err; rclone copy $O $RO -q --include "wmshape3_table.*" --include wmshape3_curves.json; }
if [ $STG = s2 ] || [ $STG = s3 ]; then                                                                      # 절 CM44 (2) 층별 첫 접촉 진단 (CPU, 읽기만)
  $PY exe/wmshape3_prof.py $O $JP $WMD/wm.pt BW4_1,V2_1,V2N_1,V3_2,QR,QB5,QB2,BE_1,V2NE_1 > $O/wmshape3_prof.md 2> $O/wmshape3_prof.err; rclone copy $O $RO -q --include "wmshape3_prof.*"
fi
log "단계 끝 $STG"
