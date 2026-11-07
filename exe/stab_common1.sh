#!/bin/bash
# 절 CM31 묶음 common1: 공통 세팅 후보 (레고 악화 가르기 · 레고 기본 판 비결정 발산 재현 · 국소 후퇴 오일러 등 후보). 잡 안, 커밋 디렉토리에서.
# 사용: bash exe/stab_common1.sh <출력> <R2> <단계>   -- 판 목록 = exe/common1_spec.txt 의 '<단계> <이름> <씬> <방식> <플래그...>' 줄.
#   씬: L 레고 (절 CM23 k1new 명령 K 1) · W wolf (절 CM25 base 명령) · F ficus · B bread (절 CM23 k1new) · M 수박 (옛 실행기, 절 CM24 (2) 판 = k1new 수박 명령 + --ue be --uf_res; 플래그로 덮음)
#   방식: d 결정적 판정 (+ 결과 불변 진단, 카드 하나씩 동시에) · n 비결정 진단 없음 (카드 하나 혼자) · nd 비결정 + 진단 · z 단독 시간 (카드 하나에 차례로, 비결정·진단 없음)
#   --ue·--ksplit 없는 판정 판은 판정 커널 기록만 (--ufrec --uf_res, 결과 불변).
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
base_of() { case $1 in                                                                         # stab_wolffix_reg.sh·stab_wolffix.sh 글자 그대로
  L) echo "exe/fem_an_rt_sim.py --data $SD/L000.pt --frames 80 $FBB $BA0 --newton 2 --cg 4 $TL --fp_interp --tag L000 --K 1";;
  W) echo "exe/fem_an_rt_sim.py --data $AG/wolf.pt --frames 50 $FBB --pbc --mlow 1e-3 $BA0 --cell 2 --newton 2 --cg 5 $TL --fp_interp --tag wolf --K 1";;
  F) echo "exe/fem_an_rt_sim.py --data $FI --frames 125 --floor none --cfg_bc --region_density --pg_vdamp $BA0 --newton 2 --cg 24 $TL --fp_interp --tag ficus --K 1";;
  B) echo "exe/fem_an_rt_sim.py --data $AG/bread.pt --frames 300 --floor none --pbc --pg_vdamp --visc_rpic --mlow 1e-3 $BA0 --newton 2 --cg 2 $TL --fp_interp --tag bread --K 1";;
  M) echo "exe/fem_ao_rt_sim.py --data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --newton 2 --cg 5 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh --fl_bar --fl_bst 1 --y0eb --lj_interp --fp_interp --tag wm --K 1";;
esac; }
dg_of() { local s=$1 f=" $2 " x
  case $s in L) x="--det --hgt --fhash --pdiag --qjd --bar_cnt";; W) x="--det --hgt --fhash --bar_cnt";; F) x="--det --pdiag --qjd --fhash --dump";; B) x="--det --pdiag --qjd --fhash";; M) x="--bar_cnt";; esac
  case "$f" in *" --ue "*|*" --ksplit "*) ;; *) [ $s != M ] && x="$x --ufrec --uf_res";; esac
  echo "$x"; }
log() { echo "[$(date '+%F %T')] $*" >> $O/common1_$STG.log; rclone copyto $O/common1_$STG.log $RO/common1_$STG.log -q; }
up() { rclone copy $O/$1 $RO/$1 -q --transfers 8 --s3-upload-concurrency 16 --s3-chunk-size 64M; }
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
  local n=$1 s=$2 md=$3 g=$4 x=$5 c dg=""
  [ -f $O/$n/summary.json ] && return
  case $md in d) dg="$(dg_of $s "$x")";; nd) dg="$(dg_of $s "$x" | sed 's/--det //')";; esac
  c="$(base_of $s) $dg $x"
  rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1
  log "$n 카드 $g 시작"
  CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_end.txt 2>&1
  [ $s = F ] && [ -f $O/$n/pred.pt ] && $PY exe/fem_p_period.py --data $FI --pred $O/$n/pred.pt > $O/$n/period.txt 2>&1
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(tail -2 $O/$n/period.txt 2>/dev/null | tr '\n' ' ') $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n; }
SPEC=$(grep -v '^#' exe/common1_spec.txt | awk -v s=$STG '$1==s')
# ---- 판정·비결정 판 (카드마다 하나)
while read -r st n s md x; do
  [ -z "$n" ] && continue; [ $md = z ] && continue
  ( g=$(claim_wait $BASHPID); run $n $s $md $g "$x"; rm -f $CL/$g ) & sleep 3
done <<< "$SPEC"
wait
log "판정 판 끝 $STG"
# ---- 단독 시간 (카드 하나, 차례로, a·b)
if echo "$SPEC" | awk '$4=="z"' | grep -q .; then
  ZG=$(claim_wait $$); log "단독 시간 카드 $ZG"
  for r in a b; do
    while read -r st n s md x; do
      [ "$md" = z ] || continue; run ${n}_$r $s z $ZG "$x"
    done <<< "$SPEC"
  done
  rm -f $CL/$ZG
fi
log "단계 끝 $STG"
