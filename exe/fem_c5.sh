#!/bin/bash
# 절 C 5 차 (디스크 /home 100 % -- 큰 덤프 없이): (a) 표·궤적 대조 (q_*, 기존 pred.pt 읽기만) (b) 검산 c5 N256 80 프레임 (GPU5, --cchk)
#   (c) 완주율 반복 (비측정 GPU5·6, --no_dump): 기존·--cfast·--cfuse_mode 2 c5 N256 각 6 회 (d) 단독 측정: --cfuse_mode 2 lego c5 N256·c2 N192·c5 N320, ficus N640, 구간 이벤트
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
python exe/fem_c_table.py --ref q_c5_n256_base $W/lbours/fem_c_lego/q_c5_* > /tmp/c_tq5.md 2>&1
python exe/fem_c_table.py --ref q_c2_n192_base $W/lbours/fem_c_lego/q_c2_* > /tmp/c_tq2.md 2>&1
cat /tmp/c_tq5.md /tmp/c_tq2.md; rclone copy /tmp/c_tq5.md r2:storage/result/anchorflow/lbours/fem_c_lego/ -q; rclone copy /tmp/c_tq2.md r2:storage/result/anchorflow/lbours/fem_c_lego/ -q
( $R k_c5_n256_chk 5 $LB --lim_sr 5 --N 256 --cfast --no_graph --cchk < /dev/null > /dev/null 2>&1; echo "[검산] $(grep -a '절 C 검산' $W/lbours/fem_c_lego/k_c5_n256_chk/v_fem.log)"
  for r in 1 2 3; do for v in base:"" cf:--cfast; do k=${v%%:*}; f=${v#*:}; $R x_c5_n256_${k}_r$r 5 $LB --lim_sr 5 --N 256 $f < /dev/null > /dev/null 2>&1; done; done ) &
( for r in 1 2 3 4 5 6; do $R x_c5_n256_fu2_r$r 6 $LB --lim_sr 5 --N 256 --cfuse --cfuse_mode 2 < /dev/null > /dev/null 2>&1; done
  for r in 4 5 6; do for v in base:"" cf:--cfast; do k=${v%%:*}; f=${v#*:}; $R x_c5_n256_${k}_r$r 6 $LB --lim_sr 5 --N 256 $f < /dev/null > /dev/null 2>&1; done; done ) &
F2="--cfuse --cfuse_mode 2"
$T y_c5_n256_fu2_r2 0 $LB --lim_sr 5 --N 256 $F2 < /dev/null > /dev/null 2>&1
$T y_c2_n192_fu2_r2 0 $LB --lim_sr 2 --N 192 $F2 < /dev/null > /dev/null 2>&1
$T y_c5_n256_fu2_sec 0 $LB --lim_sr 5 --N 256 $F2 --secprof < /dev/null > /dev/null 2>&1
$T y_c5_n320_fu2_r1 0 $LB --lim_sr 5 --N 320 $F2 < /dev/null > /dev/null 2>&1
$T y_c5_n320_fu2_r2 0 $LB --lim_sr 5 --N 320 $F2 < /dev/null > /dev/null 2>&1
$TF y_f_c2_n640_fu2_r1 0 $FB --lim_sr 2 --N 640 $F2 < /dev/null > /dev/null 2>&1
$TF y_f_c2_n640_fu2_r2 0 $FB --lim_sr 2 --N 640 $F2 < /dev/null > /dev/null 2>&1
$TF y_f_c2_n640_cf_r2 0 $FB --lim_sr 2 --N 640 --cfast < /dev/null > /dev/null 2>&1
wait
python exe/fem_c_table.py --ref x_c5_n256_base $W/lbours/fem_c_lego/[tuvwxy]_* > /tmp/c_t5.md 2>&1
python exe/fem_c_table.py $W/lbours/fem_c_ficus/[uy]_* > /tmp/c_t5f.md 2>&1
cat /tmp/c_t5.md /tmp/c_t5f.md
cp /tmp/c_t5.md $W/lbours/fem_c_lego/table5.md; cp /tmp/c_t5f.md $W/lbours/fem_c_ficus/table5.md
rclone copy /tmp/c_t5.md r2:storage/result/anchorflow/lbours/fem_c_lego/ -q; rclone copy /tmp/c_t5f.md r2:storage/result/anchorflow/lbours/fem_c_ficus/ -q
echo "C5_DONE $(date +%T) $(df -h /home | tail -1)"
