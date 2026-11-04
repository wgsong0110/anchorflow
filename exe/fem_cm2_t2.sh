#!/bin/bash
# 절 CM2 진단 t2 (커밋 $1, 카드 1-6, 결정적, 비측정): 새 공통 묶음에서 bread·wolf 오차가 커진 원인 -- 묶음 인자 하나씩 뺀 판 (2x10).
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_cm2 XDG_CACHE_HOME=/dev/shm/dkta/cache
PY=/home/dkta/.conda/envs/af/bin/python
NM=cm2t2; R=/dev/shm/dkta/lbours/fem_cm/$NM; RR=r2:storage/result/anchorflow/lbours/fem_cm/$NM; mkdir -p $R
source exe/fem_cm2_scn.sh
T="--tolN 1e-2 --tolP 1e-1 --det"; G8="--ga hht --rho 0.8 --K 2 --newton 2 --cg 10"
run() { local G=$1 V=$2; shift 2; mkdir -p $R/$V
  echo "카드 $G $C $(date '+%T') $*" > $R/$V/run_info.txt
  CUDA_VISIBLE_DEVICES=$G timeout 1500 $PY -u exe/fem_an_rt_sim.py --out $R/$V --tag $V "$@" < /dev/null > $R/$V/log.txt 2>&1
  echo "rc=$? $(date '+%T')" >> $R/$V/run_info.txt
  echo "$V $(grep -a '롤아웃\|rror\|발산' $R/$V/log.txt | tail -2 | cut -c1-200 | tr '\n' ' ')" >> $R/done.txt
  rclone copy $R/$V $RR/$V -q --exclude pred.pt; rclone copy $R/done.txt $RR -q; }
cm_scene bread; BR="$DA"; cm_scene wolf; WO="$DA"
( run 1 bread_nooct ${BR//--gq_oct 0.1/} $G8 $T; run 1 bread_noy0 ${BR//--ga_y0ut/} $G8 $T ) &
( run 2 bread_jac $BR --mg_levels 1 $G8 $T; run 2 bread_c20 $BR $G8 $T --cg 20 ) &
( run 3 wolf_nooct ${WO//--gq_oct 0.1/} $G8 $T; run 3 wolf_noy0 ${WO//--ga_y0ut/} $G8 $T ) &
( run 4 wolf_stick ${WO//--fl_bar --fl_bst 1/} $G8 $T; run 4 wolf_jac $WO --mg_levels 1 $G8 $T ) &
( run 5 wolf_c20 $WO $G8 $T --cg 20; run 5 wolf_r7k3 $WO $G8 $T --rho 0.7 --K 3 ) &
( run 6 bread_r7k3 $BR $G8 $T --rho 0.7 --K 3 ) &
wait
echo "[끝 $NM] $(date '+%F %T')" >> $R/done.txt; rclone copy $R/done.txt $RR -q
