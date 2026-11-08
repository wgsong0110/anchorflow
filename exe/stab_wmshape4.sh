#!/bin/bash
# 절 CM46 묶음 wmshape4: V2NF 발산 원인 (D2) + V2NF·V2N + W̄ (절 CM45 --e3w 1) + 첫 접촉 진단 D4·후보 T (lib/anchorflow/fem_wmshape4.py). 잡 안, 커밋 디렉토리에서.
# exe/stab_wmshape3.sh 와 같은 틀 + 감싸개 exe/wmshape4_wrap.py ('@w4*' -> '@e3*' -> '@e2*' 순서로 앞에).
# 사용: bash exe/stab_wmshape4.sh <출력> <R2> <단계>   -- 판 목록 = exe/wmshape4_spec.txt
O=$1; RO=$2; STG=$3; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
WMD=/tmp/dkta/ag_data; mkdir -p $WMD
[ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
JP=/dev/shm/dkta/ag_data/gt_jp; mkdir -p $JP
[ $(ls $JP | wc -l) -ge 101 ] || rclone copy r2:storage/result/anchorflow/lbours/fem_cm/wmJd/gt_jp $JP -q --transfers 16
WMB="exe/fem_ao_rt_sim.py --data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --newton 2 --cg 5 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh --fl_bar --fl_bst 1 --y0eb --lj_interp --fp_interp --tag wm --K 1 --gq_oct 0 --ws_bar 4 --ws_fi"
BW4="--y0loc 3 --ue be --uf_res --newton 6 --xit_from 2 --en_mo --ls0_keep --ue_loc 50 --sp2 31 --ksplit 3 --ks_big --xg 1 --wrx 3 --wrx_col 1 --rm_sub 16 --ws2_fc 20,20 --ws2_fcn 2 --ws2_win 4"   # 절 CM41 BW4 (절 CM44 바탕)
V2N="$BW4 --ws3 1 --ws3_win 1 --ws3_lj 1"; V2NF="$V2N --ws3_pf 1"                     # 절 CM44 V2N·V2NF 글자 그대로
base_of() { case $1 in M3) echo "${WMB/--cell 1.5/--cell 3}";; esac; }
nw() { case " $1 " in *" --newton "*) echo "$1" | sed 's/.*--newton \([0-9]*\).*/\1/';; *) echo "";; esac; }
SVF=3,4,5,6,8,10,15,20,30,50,100
dg_of() { case $1 in r) echo "--erec --wsh_rec 30 --wsh_sv $SVF --ws2 1";; z) echo "--erec";; esac; }
log() { echo "[$(date '+%F %T')] $*" >> $O/wmshape4_$STG.log; rclone copyto $O/wmshape4_$STG.log $RO/wmshape4_$STG.log -q; }
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
  local n=$1 s=$2 md=$3 g=$4 x=" $5 " c b w=""
  [ -f $O/$n/summary.json ] && return
  x="${x// V2NF / $V2NF }"; x="${x// V2N / $V2N }"
  for p in w4t w4dg e3w e2g; do                                                          # 감싸개 인자 (이 순서로 앞에)
    v="$(echo "$x" | grep -o "@$p [0-9.,e+-]*" | head -1 | awk '{print $2}')"; [ -n "$v" ] && w="$w --$p $v"
  done
  x="$(echo "$x" | sed 's/@[a-z0-9_]* [0-9.,e+-]*//g')"
  b="$(base_of $s)"; [ -n "$(nw "$x")" ] && b="${b/--newton 2/--newton $(nw "$x")}" && x="$(echo "$x" | sed 's/--newton [0-9]*//g')"
  c="exe/wmshape4_wrap.py $w $b $(dg_of $md) $x"
  rm -rf $O/$n; mkdir -p $O/$n
  local pre=""; [ $md = r ] && pre="AF_E2_DG=1"
  echo "$pre $PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1
  log "$n 카드 $g 시작"
  env $pre CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_end.txt 2>&1
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n; }
SPEC=$(grep -v '^#' exe/wmshape4_spec.txt | awk -v s=$STG '$1==s')
mkdir -p $O/gt; [ -s $O/gt/gt_wmshape.json ] || rclone copyto r2:storage/result/anchorflow/lbours/wmshape/gt/gt_wmshape.json $O/gt/gt_wmshape.json -q
[ -s $O/gt/gt_wmshape2.json ] || rclone copyto r2:storage/result/anchorflow/lbours/wmshape2/gt/gt_wmshape2.json $O/gt/gt_wmshape2.json -q
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
if [ $STG = s0 ]; then                                                                      # 연기: 모든 판이 5 프레임 넘게, 진단 기록 있음
  OK=1; while read -r st n s md x; do [ -z "$n" ] && continue
    $PY -c "import json,sys; s=json.load(open('$O/$n/summary.json')); sys.exit(0 if s['frames_done'] >= 5 and not s.get('diverged') else 1)" 2>/dev/null || OK=0
    grep -aq '"ws3": {' $O/$n/fem.jsonl 2>/dev/null || OK=0
    [ -s $O/$n/e2.jsonl ] || OK=0
    case "$x" in *@w4dg*) grep -q '"tag": "fin"' $O/$n/w4.jsonl 2>/dev/null || OK=0;; esac
    case "$x" in *@w4t*) grep -aq '후보 T' $O/$n/log.txt || OK=0;; esac
    case "$x" in *@e3w*) grep -aq 'W̄ 1' $O/$n/log.txt || OK=0;; esac
    [ -s $O/$n/w4_err.txt ] && OK=0
  done <<< "$SPEC"
  log "[연기] OK=$OK"; [ $OK = 1 ] || { echo "연기 실패 $(date '+%F %T')" > $O/wmshape4.FAIL; rclone copy $O $RO -q --include wmshape4.FAIL; exit 1; }
fi
[ $STG = s0 ] || { $PY exe/wmshape4_tab.py $O $JP > $O/wmshape4_table.md 2> $O/wmshape4_table.err; rclone copy $O $RO -q --include "wmshape4_table.*" --include wmshape4_curves.json; }
if [ $STG != s0 ]; then                                                                     # 층별 첫 접촉 (절 CM44 exe/wmshape3_prof.py 그대로, 읽기만)
  NS=$(ls -d $O/*/wsh 2>/dev/null | xargs -n1 dirname | xargs -n1 basename | grep -E '^(V2NF_1|V2NFW_1|V2NW_1|dV2N_1|dV2N_2|dV2NW_1|V2NFWT_1|V2NFWT_2|V2NWT_1|V2NWT_2|dV2NWT_1|dV2NWT_2)$' | tr '\n' ',' | sed 's/,$//')
  [ -n "$NS" ] && $PY exe/wmshape3_prof.py $O $JP $WMD/wm.pt $NS > $O/wmshape4_prof.md 2> $O/wmshape4_prof.err; rclone copy $O $RO -q --include "wmshape4_prof.*"
fi
log "단계 끝 $STG"
