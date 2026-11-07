#!/bin/bash
# 절 CM37 묶음 common3: 공통 후보 C (BG + 무너진 걸음만 쪼개기 --ksplit 3 --ks_big) + BG 시간 증가 고침 (--xg 나눈 그래프, 결과 불변). 잡 안, 커밋 디렉토리에서.
# 사용: bash exe/stab_common3.sh <출력> <R2> <단계>   -- 판 목록 = exe/common3_spec.txt 의 '<단계> <이름> <씬> <방식> <플래그...>' 줄.
#   씬: L 레고 (절 CM23 k1new 명령 K 1) · W wolf (절 CM25 base) · F ficus · B bread · M15·M3 수박 (옛 실행기, 절 CM33 (2) 판 바탕 = k1new 수박 명령 + --gq_oct 0 --ws_bar 4 --ws_fi, 셀 1.5·3; 쪼개기 끔)
#   방식: d 결정적 판정 (+ 결과 불변 진단, 카드 하나씩 동시에) · nd 비결정 + 진단 · n 비결정 진단 없음 · a 수박 진단 (--enrec 몫 분해, 결과 불변) · b 수박 --erec 만
#         z 단독 시간 (카드 하나에 차례로, 비결정·진단 없음; 수박은 --erec 만 = 손상·에너지 기록) -- z 는 판정 판이 모두 끝난 뒤.
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
WMB="exe/fem_ao_rt_sim.py --data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --newton 2 --cg 5 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh --fl_bar --fl_bst 1 --y0eb --lj_interp --fp_interp --tag wm --K 1 --gq_oct 0 --ws_bar 4 --ws_fi"
base_of() { case $1 in                                                                         # stab_common1.sh 글자 그대로 (수박은 절 CM33 stab_wmen.sh 의 (2) 판 바탕)
  L) echo "exe/fem_an_rt_sim.py --data $SD/L000.pt --frames 80 $FBB $BA0 --newton 2 --cg 4 $TL --fp_interp --tag L000 --K 1";;
  W) echo "exe/fem_an_rt_sim.py --data $AG/wolf.pt --frames 50 $FBB --pbc --mlow 1e-3 $BA0 --cell 2 --newton 2 --cg 5 $TL --fp_interp --tag wolf --K 1";;
  F) echo "exe/fem_an_rt_sim.py --data $FI --frames 125 --floor none --cfg_bc --region_density --pg_vdamp $BA0 --newton 2 --cg 24 $TL --fp_interp --tag ficus --K 1";;
  B) echo "exe/fem_an_rt_sim.py --data $AG/bread.pt --frames 300 --floor none --pbc --pg_vdamp --visc_rpic --mlow 1e-3 $BA0 --newton 2 --cg 2 $TL --fp_interp --tag bread --K 1";;
  M15) echo "$WMB";;
  M3) echo "${WMB/--cell 1.5/--cell 3}";;
esac; }
nw() { case " $1 " in *" --newton "*) echo "$1" | sed 's/.*--newton \([0-9]*\).*/\1/';; *) echo "";; esac; }
dg_of() { local s=$1 md=$2 f=" $3 " x
  case $s in L) x="--det --hgt --fhash --pdiag --qjd --bar_cnt";; W) x="--det --hgt --fhash --bar_cnt";; F) x="--det --pdiag --qjd --fhash --dump";; B) x="--det --pdiag --qjd --fhash";; M*) x="";; esac
  case $s in M*) case $md in a) x="--qjd --jdiag --bar_cnt --erec --ufrec --nrec --enrec 100";; b|z) x="--erec";; esac; echo "$x"; return;; esac
  case $md in nd) x="${x/--det /}"; [ $s = W ] && x="$x --dump --dump_part";; z|n) x="";; esac   # wolf 비결정: 입자 이탈 진단용 덤프 (결과 불변)
  [ $md = z ] || [ $md = n ] || case "$f" in *" --ue "*|*" --ksplit "*) ;; *) x="$x --ufrec --uf_res";; esac
  echo "$x"; }
log() { echo "[$(date '+%F %T')] $*" >> $O/common3_$STG.log; rclone copyto $O/common3_$STG.log $RO/common3_$STG.log -q; }
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
  b="$(base_of $s)"; [ -n "$(nw "$x")" ] && b="${b/--newton 2/--newton $(nw "$x")}" && x="$(echo "$x" | sed 's/--newton [0-9]*//')"   # --newton 은 바탕 값을 바꿈 (중복 인자 없이)
  c="$b $(dg_of $s $md "$x") $x"
  rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1
  log "$n 카드 $g 시작"
  CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_end.txt 2>&1
  if [ $s = W ] && [ -f $O/$n/pred.pt ]; then                                               # 입자 이탈·조밀 상자 (exe/speed1_bbox.py, 절 CM34 진단)
    $PY exe/speed1_bbox.py $AG/wolf.pt $(grep -aom1 "셀 [0-9.]* m" $O/$n/log.txt | awk '{print $2}') $O/$n/pred.pt > $O/$n/bbox.txt 2>&1; fi
  [ $s = F ] && [ -f $O/$n/pred.pt ] && $PY exe/fem_p_period.py --data $FI --pred $O/$n/pred.pt > $O/$n/period.txt 2>&1
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(tail -2 $O/$n/period.txt 2>/dev/null | tr '\n' ' ') $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n; }
SPEC=$(grep -v '^#' exe/common3_spec.txt | awk -v s=$STG '$1==s')
# ---- 판정·비결정·진단 판 (카드마다 하나)
while read -r st n s md x; do
  [ -z "$n" ] && continue; [ $md = z ] && continue; [ -f $O/$n/summary.json ] && continue
  while [ $(jobs -rp | wc -l) -ge $NG ]; do sleep 5; done
  ( me=$BASHPID; g=$(claim_wait $me); run $n $s $md $g "$x"; rm -f $CL/$g ) & sleep 3
done <<< "$SPEC"
wait
log "판정 판 끝 $STG"
# ---- 단독 시간 (카드 하나, 차례로, a·b; 시작 때 그 카드 다른 계산 프로세스 0)
if echo "$SPEC" | awk '$4=="z"' | grep -q .; then
  ZG=$(claim_wait $$); log "단독 시간 카드 $ZG"
  for r in a b; do
    while read -r st n s md x; do
      [ "$md" = z ] || continue; run ${n}_$r $s z $ZG "$x"
    done <<< "$SPEC"
  done
  rm -f $CL/$ZG
fi
if [ $STG = s0 ]; then                                                                      # 연기: 모든 판이 5 프레임 넘게 끝나야 (컴파일·훅 확인)
  OK=1; while read -r st n s md x; do [ -z "$n" ] && continue
    $PY -c "import json,sys; s=json.load(open('$O/$n/summary.json')); sys.exit(0 if s['frames_done'] >= 5 and not s.get('diverged') else 1)" 2>/dev/null || OK=0
  done <<< "$SPEC"
  log "[연기] OK=$OK"; [ $OK = 1 ] || { echo "연기 실패 $(date '+%F %T')" > $O/common3.FAIL; rclone copy $O $RO -q --include common3.FAIL; exit 1; }
fi
$PY exe/common3_tab.py $O > $O/common3_table.md 2> $O/common3_table.err; rclone copy $O $RO -q --include "common3_table.*"
log "단계 끝 $STG"
