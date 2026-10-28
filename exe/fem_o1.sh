#!/bin/bash
# 절 O 1 차 (사용자 10-28 "2번"): lag 명시적 판 + 속도 단위 바닥 (--floor vel). 모두 GPU0 단독 순차 측정 (fem_ex_run.sh 가 7 장 빌 때까지 대기).
# (0) 스모크 5 프레임 (1) lego L000 80 프레임 N 스윕 (고착 분배 frac/full, 비관통 끔, 아핀 전달) (2) 계측 (torch.profiler) (3) ficus 125 프레임 같은 코드 경로 (바닥 없음)
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; R="bash exe/fem_ex_run.sh"
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
LG="--mode lag --opt --mass tet --no_dump --floor vel"
export EX_ROOT=fem_o_lego
$R o_smoke 0 $LG --N 96 --frames 5
[ -f $W/lbours/fem_o_lego/o_smoke/EX_DONE ] || { echo "[스모크 실패]"; tail -30 $W/lbours/fem_o_lego/o_smoke/v_fem.log; echo O1_FAIL; exit 1; }
for N in 32 48 64 80 96 128 192; do $R o_fr_n$N 0 $LG --N $N; done
for N in 48 64 96 128; do $R o_fu_n$N 0 $LG --N $N --vfl_mode full; done
$R o_fr_np0_n96 0 $LG --N 96 --vfl_np 0
for N in 64 96 128; do $R o_fr_aff_n$N 0 $LG --N $N --affine; done
$R o_fu_aff_n96 0 $LG --N 96 --vfl_mode full --affine
$R o_prof_n96 0 $LG --N 96 --frames 4 --tprof 3 --no_graph
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
PHYS="--data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
export EX_ROOT=fem_o_ficus
for N in 512 640 768; do $R o_fi_n$N 0 --mode lag --opt --mass tet --no_dump $PHYS --N $N; done
O=$W/lbours/fem_o_lego/o1_summary.txt; : > $O
for d in $W/lbours/fem_o_lego/o_* $W/lbours/fem_o_ficus/o_*; do echo "== $(basename $d) $(grep -a "롤아웃\]" $d/v_fem.log | cut -c1-260) $(grep -a "발산\]" $d/v_fem.log | head -1 | cut -c1-100)" >> $O; done
cat $O; rclone copy $O r2:storage/result/anchorflow/lbours/fem_o_lego/ -q
echo O1_DONE
