#!/bin/bash
# 절 CM2 시험 t1 (커밋 $1, 카드 1-6, 결정적): 회귀 (탄성 판 옛 인자 = cm1 c3_L000 해시), 재질 판 새 실행기로 옛 설정 (블록 야코비·가우스 점 고착) 이 implicit-wolf 값 근처인지, 새 공통 설정 시험.
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_cm2 XDG_CACHE_HOME=/dev/shm/dkta/cache
PY=/home/dkta/.conda/envs/af/bin/python
NM=${CM_NM:-cm2t1}; R=/dev/shm/dkta/lbours/fem_cm/$NM; RR=r2:storage/result/anchorflow/lbours/fem_cm/$NM; mkdir -p $R
source exe/fem_cm2_scn.sh
for t in L000 ficus wolf bread drop1; do cm_data $t; done
T="--tolN 1e-2 --tolP 1e-1 --det"
BO="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2"
run() { local G=$1 V=$2; shift 2; mkdir -p $R/$V
  echo "카드 $G $C $(date '+%T') $*" > $R/$V/run_info.txt
  CUDA_VISIBLE_DEVICES=$G timeout ${TMO:-1500} $PY -u exe/fem_an_rt_sim.py --out $R/$V --tag $V "$@" < /dev/null > $R/$V/log.txt 2>&1
  echo "rc=$? $(date '+%T')" >> $R/$V/run_info.txt
  case $V in *ficus*) [ -f $R/$V/pred.pt ] && $PY exe/fem_p_period.py --data $FI --pred $R/$V/pred.pt > $R/$V/period.txt 2>&1;; esac
  case $V in *drop1*) [ -f $R/$V/pred.pt ] && $PY exe/fem_ak_drop_ana.py $DD/init.pt mpm:$DD/mpm/x240.pt:0.0041666667 $V:$R/$V/pred.pt:0.0166666667 > $R/$V/drop_ana.txt 2>&1;; esac
  echo "$V $(grep -a '롤아웃\|rror\|발산' $R/$V/log.txt | tail -2 | cut -c1-200 | tr '\n' ' ') $(grep -o '"x_sha1": "[0-9a-f]*' $R/$V/summary.json 2>/dev/null)" >> $R/done.txt
  rclone copy $R/$V $RR/$V -q --exclude pred.pt; rclone copy $R/done.txt $RR -q; }
G8="--ga hht --rho 0.8 --K 2"
( run 1 reg_L000 --data $SD/L000.pt --frames 80 --floor vel --fl_bar --fl_bst 1 $BO $G8 --newton 2 --cg 3 $T
  cm_scene L000; run 1 y5_L000 $DA $G8 --newton 2 --cg 5 $T
  cm_scene drop1; run 1 y5_drop1 $DA $G8 --newton 2 --cg 5 $T ) &
( cm_scene ficus; run 2 y10_ficus $DA $G8 --newton 2 --cg 10 $T
  cm_scene 0016; run 2 y5_0016 $DA $G8 --newton 2 --cg 5 $T ) &
( run 3 v_wolf --data $AG/wolf.pt --frames 50 --cell 2 --floor vel --pbc --mlow 1e-3 --ga hht --rho 0.7 --ga_y0ut --K 3 --newton 2 --cg 5 $T
  cm_scene 0019; run 3 y5_0019 $DA $G8 --newton 2 --cg 5 $T ) &
( run 4 v_bread --data $AG/bread.pt --frames 300 --floor none --pbc --pg_vdamp --visc_rpic --mlow 1e-3 --ga hht --rho 0.7 --ga_y0ut --K 3 --newton 2 --cg 5 $T ) &
( cm_scene wolf; run 5 y10_wolf $DA $G8 --newton 2 --cg 10 $T ) &
( cm_scene bread; run 6 y10_bread $DA $G8 --newton 2 --cg 10 $T ) &
wait
echo "[끝 $NM] $(date '+%F %T')" >> $R/done.txt; rclone copy $R/done.txt $RR -q
