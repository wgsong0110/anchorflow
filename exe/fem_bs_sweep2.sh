#!/bin/bash
# 절 M 스윕 2: 셀 대표 적분점 (힘·변형 같은 연산자로 고친 판) N 스윕, 입자 적분점 경계 보충, 입자 적분점 최적화 전/후·결과 불변 (기준 판 double, 반복, 그래프 끔). GPU0 단독 순차.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; R="bash exe/fem_ex_run.sh"
B="--interp bspline2 --mode sub --no_dump"
PHYS="--data $W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
export EX_ROOT=fem_bs_lego
for N in 48 64 96 128; do $R s2_c8_n$N 0 $B --opt --quad cell --K 8 --N $N; done
for N in 48 64 96 128; do $R s2_c1_n$N 0 $B --opt --quad cell --K 1 --N $N; done
$R s2_p_n88 0 $B --opt --N 88
$R s2_p_n96_p1 0 $B --opt --N 96 --dump_pred
$R s2_p_n96_p2 0 $B --opt --N 96 --dump_pred
$R s2_p_n96_nog 0 $B --opt --N 96 --dump_pred --no_graph
$R s2_p_n96_base 0 $B --N 96 --dump_pred
echo SW2_LEGO_DONE
export EX_ROOT=fem_bs_ficus
for N in 256 384 512 768; do $R s2_c8_n$N 0 $B --opt --quad cell --K 8 --N $N $PHYS; done
for N in 256 384 512 768; do $R s2_c1_n$N 0 $B --opt --quad cell --K 1 --N $N $PHYS; done
$R s2_p_n448 0 $B --opt --N 448 $PHYS
$R s2_p_n512_p1 0 $B --opt --N 512 --dump_pred $PHYS
$R s2_p_n512_p2 0 $B --opt --N 512 --dump_pred $PHYS
$R s2_p_n512_nog 0 $B --opt --N 512 --dump_pred --no_graph $PHYS
$R s2_p_n512_base 0 $B --N 512 --dump_pred $PHYS
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af
O=$W/lbours/fem_bs_lego/cmp2.txt; : > $O
for sh in lego ficus; do
  L=$W/lbours/fem_bs_$sh; [ $sh = lego ] && n=n96 || n=n512
  for pr in "p1 p2 같은설정반복" "p1 nog 그래프_vs_없음" "base nog 기준판(double·따로·서브스텝격자)_vs_최적화(그래프없음)" "base p1 기준판_vs_최종"; do
    set -- $pr; echo "== $sh s2_p_${n}_$1 vs s2_p_${n}_$2 ($3)" >> $O
    python -u exe/fem_rt_cmp.py $L/s2_p_${n}_$1 $L/s2_p_${n}_$2 2>&1 | tail -1 >> $O
  done
done
cat $O; rclone copy $O r2:storage/result/anchorflow/lbours/fem_bs_lego/ -q
for d in $W/lbours/fem_bs_lego/s2_* $W/lbours/fem_bs_ficus/s2_*; do echo "== $(basename $d) $(grep -a "롤아웃\]" $d/v_fem.log | cut -c30-230) $(grep -a "발산\]" $d/v_fem.log|head -1|cut -c1-90)"; done
echo SW2_DONE
