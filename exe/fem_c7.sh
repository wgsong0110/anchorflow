#!/bin/bash
# 절 C 7 차: (6) --sr_ef (힘 커널의 ‖A-I‖_F² 재사용) -- 검산 (GPU5, --cfast --sr_ef --cchk c5 N256 80 프레임), 단독 측정 (융합 2 + sr_ef: c5 N256·c2 N192·c5 N320·ficus),
#            비측정 오차 (GPU6): c5 N320·N352 융합 2 + sr_ef 각 3 회
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T) $(df -h /home | tail -1)"
LB="--mode lag --opt --mass tet --floor vel --rstat --no_dump"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
FB="--mode lag --opt --mass tet --data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --no_dump"
R="env EX_ROOT=fem_c_lego bash exe/fem_c_run.sh"
T="env EX_TIMED=1 EX_ROOT=fem_c_lego bash exe/fem_c_run.sh"
TF="env EX_TIMED=1 EX_ROOT=fem_c_ficus bash exe/fem_c_run.sh"
F3="--cfuse --cfuse_mode 2 --sr_ef"
$R s_c5_ef 5 $LB --lim_sr 5 --N 256 $F3 --frames 3 < /dev/null > /dev/null 2>&1
echo "[연기] $(tail -1 $W/lbours/fem_c_lego/s_c5_ef/v_fem.log | cut -c1-200)"; [ -f $W/lbours/fem_c_lego/s_c5_ef/EX_DONE ] || { tail -20 $W/lbours/fem_c_lego/s_c5_ef/v_fem.log; echo C7_FAIL; exit 1; }
( $R k_c5_n256_ef_chk 5 $LB --lim_sr 5 --N 256 --cfast --sr_ef --no_graph --cchk < /dev/null > /dev/null 2>&1; echo "[검산] $(grep -a '절 C 검산' $W/lbours/fem_c_lego/k_c5_n256_ef_chk/v_fem.log)"
  for r in 1 2 3; do $R z_c5_n256_ef_r$r 5 $LB --lim_sr 5 --N 256 $F3 < /dev/null > /dev/null 2>&1; done ) &
( for r in 1 2 3; do for n in 320 352; do $R z_c5_n${n}_ef_r$r 6 $LB --lim_sr 5 --N $n $F3 < /dev/null > /dev/null 2>&1; done; done ) &
for r in 1 2; do
  $T zt_c5_n256_ef_r$r 0 $LB --lim_sr 5 --N 256 $F3 < /dev/null > /dev/null 2>&1
  $T zt_c2_n192_ef_r$r 0 $LB --lim_sr 2 --N 192 $F3 < /dev/null > /dev/null 2>&1
  $T zt_c5_n320_ef_r$r 0 $LB --lim_sr 5 --N 320 $F3 < /dev/null > /dev/null 2>&1
  $TF zt_f_c2_n640_ef_r$r 0 $FB --lim_sr 2 --N 640 $F3 < /dev/null > /dev/null 2>&1
done
$T zt_c5_n256_ef_sec 0 $LB --lim_sr 5 --N 256 $F3 --secprof < /dev/null > /dev/null 2>&1
wait
python exe/fem_c_table.py $W/lbours/fem_c_lego/z*_* > /tmp/c_t7.md 2>&1
python exe/fem_c_table.py $W/lbours/fem_c_ficus/zt_* > /tmp/c_t7f.md 2>&1
cat /tmp/c_t7.md /tmp/c_t7f.md
rclone copy /tmp/c_t7.md r2:storage/result/anchorflow/lbours/fem_c_lego/ -q; rclone copy /tmp/c_t7f.md r2:storage/result/anchorflow/lbours/fem_c_ficus/ -q
echo "C7_DONE $(date +%T) $(df -h /home | tail -1)"
