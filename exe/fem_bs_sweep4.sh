#!/bin/bash
# 절 M 4 차 (조정자 10-28: 60 fps·오차 10 % 목표 대비 병목 수치): 커널별 GPU 시간 (torch.profiler, 프레임 3, 그래프 끔), 입자 정렬 (--psort) 속도·결과 불변. GPU0 단독 순차.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; R="bash exe/fem_ex_run.sh"
B="--interp bspline2 --mode sub --opt"
PHYS="--data $W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
export EX_ROOT=fem_bs_lego
$R s4_prof_n96 0 $B --N 96 --no_dump --no_graph --tprof 3 --frames 5
$R s4_sort_n96_p1 0 $B --N 96 --no_dump --dump_pred --psort
$R s4_sort_n96_p2 0 $B --N 96 --no_dump --dump_pred --psort
$R s4_sort_n88 0 $B --N 88 --no_dump --psort
$R s4_sortprof_n96 0 $B --N 96 --no_dump --no_graph --tprof 3 --frames 5 --psort
export EX_ROOT=fem_bs_ficus
$R s4_prof_n512 0 $B --N 512 --no_dump --no_graph --tprof 3 $PHYS --frames 5
$R s4_sort_n512_p1 0 $B --N 512 --no_dump --dump_pred --psort $PHYS
$R s4_sort_n512_p2 0 $B --N 512 --no_dump --dump_pred --psort $PHYS
$R s4_sort_n448 0 $B --N 448 --no_dump --psort $PHYS
$R s4_sortprof_n512 0 $B --N 512 --no_dump --no_graph --tprof 3 --frames 5 --psort $PHYS
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af
O=$W/lbours/fem_bs_lego/cmp4.txt; : > $O
for pr in "lego s4_sort_n96_p1 s4_sort_n96_p2" "lego s2_p_n96_p1 s4_sort_n96_p1" "ficus s4_sort_n512_p1 s4_sort_n512_p2" "ficus s2_p_n512_p1 s4_sort_n512_p1"; do
  set -- $pr; echo "== $1 $2 vs $3" >> $O; python -u exe/fem_rt_cmp.py $W/lbours/fem_bs_$1/$2 $W/lbours/fem_bs_$1/$3 2>&1 | tail -1 >> $O; done
cat $O; rclone copy $O r2:storage/result/anchorflow/lbours/fem_bs_lego/ -q
for d in $W/lbours/fem_bs_lego/s4_* $W/lbours/fem_bs_ficus/s4_*; do echo "== $(basename $d) $(grep -a "롤아웃\]" $d/v_fem.log | cut -c30-200) $(grep -a "발산\]" $d/v_fem.log|head -1|cut -c1-80)"; done
for d in $W/lbours/fem_bs_lego/s4_*prof* $W/lbours/fem_bs_ficus/s4_*prof*; do echo "== $(basename $d)"; grep -a -A 18 "Self CUDA" $d/v_fem.log | cut -c1-40,100-260 | head -20; done
echo SW4_DONE
