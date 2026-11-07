#!/bin/bash
# 절 CM25 회귀: wolf 를 무너뜨린 판 (A) = 각 씬 기본 판 (절 CM23 K 1 명령) 에서 장벽 바닥 플래그 (--fl_bar --fl_bst 1 --y0eb) 만 뺀 것 (바닥 = --floor vel 가우스 점 고착) 을 다섯 씬 같은 플래그로.
#   ficus·bread 는 바닥 없음 (--floor none) 이라 뺄 플래그가 없어 기본 판과 같은 명령 -> 절 CM23 F_K1·B_K1 재사용 (돌리지 않음).
#   판정: L_A (결정적 + --pdiag --qjd --fhash), M_A_a·M_A_b (수박 비결정 두 번, 진단 --gq_chk --nan_step --jdiag --qjd); wolf = 묶음 wolfwhy A_stick.
#   단독 시간: 진단·--det 뺀 비결정 판, 카드 하나에 차례로 (z_L_A_a·b, z_W_A_a·b, z_M_A_a·b), 그 카드·노드 계산 프로세스 수 10 초마다.
O=$1; R=$2; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
source exe/fem_cm2_scn.sh
for t in L000 wolf; do cm_data $t; done
WMD=/tmp/dkta/ag_data; mkdir -p $WMD
[ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
BA0="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2 --ga_y0ut --hspec --ncomp --ga hht --rho 0.8"
TL="--tolN 1e-2 --tolP 1e-1"
base_of() { case $1 in   # 절 CM23 base_of 에서 "--fl_bar --fl_bst 1 --y0eb" 만 뺌
  L) echo "exe/fem_an_rt_sim.py --data $SD/L000.pt --frames 80 --floor vel $BA0 --newton 2 --cg 4 $TL --fp_interp --tag L000 --K 1";;
  W) echo "exe/fem_an_rt_sim.py --data $AG/wolf.pt --frames 50 --floor vel --pbc --mlow 1e-3 $BA0 --cell 2 --newton 2 --cg 5 $TL --fp_interp --tag wolf --K 1";;
  M) echo "exe/fem_ao_rt_sim.py --data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut $TL --newton 2 --cg 5 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh --lj_interp --fp_interp --tag wm --K 1";;
esac; }
dg_of() { case $1 in L) echo "--det --pdiag --qjd --fhash";; M) echo "--gq_chk --nan_step --jdiag --qjd";; esac; }
log() { echo "[$(date '+%F %T')] $*" >> $O/reg.log; rclone copyto $O/reg.log $R/reg.log -q; }
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
run() { local n=$1 s=$2 g=$3 extra=$4 c
  [ -f $O/$n/summary.json ] && return
  c="$(base_of $s) $extra"; rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/gpu_before.txt 2>&1
  ( while sleep 10; do echo "$(date '+%T') 카드 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .) 노드 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" >> $O/$n/gpu_procs.txt; done ) & local wp=$!
  log "$n 카드 $g 시작"
  CUDA_VISIBLE_DEVICES=$g timeout 5400 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?; kill $wp 2>/dev/null
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-220) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O/$n $R/$n -q; }
# 판정 (카드마다 동시에; 잡는 pid 는 살아 있는 서브셸)
for v in "L_A L" "M_A_a M" "M_A_b M"; do set -- $v
  ( me=$BASHPID; g=$(claim_wait $me); run $1 $2 $g "$(dg_of $2)"; rm -f $CL/$g ) & sleep 5; done
wait; log "판정 끝"
# 단독 시간 (카드 하나, 차례로)
ZG=$(claim_wait $$); log "단독 시간 카드 $ZG"
for s in L W M; do for r in a b; do run z_${s}_A_$r $s $ZG ""; done; done
rm -f $CL/$ZG
log "REG_END"
