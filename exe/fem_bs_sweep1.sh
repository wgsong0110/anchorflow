#!/bin/bash
# 절 M 스윕 1: 2 차 B-스플라인 명시적 판 안정 경계 (N 스윕, 발산하면 그 프레임에서 멈춤). GPU0 단독 순차 (fem_ex_run 이 7 장 빌 때까지 대기).
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; R="bash exe/fem_ex_run.sh"
B="--interp bspline2 --mode sub --opt --no_dump"
PHYS="--data $W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
export EX_ROOT=fem_bs_lego
for N in 48 64 80 96 128 160 192; do $R s1_p_n$N 0 $B --N $N; done
for N in 20 32 48 64 96; do $R s1_p_nofl_n$N 0 $B --N $N --floor none --frames 20; done
for N in 64 96 128 192; do $R s1_c8_n$N 0 $B --quad cell --K 8 --N $N; done
for N in 64 96 128 192; do $R s1_c1_n$N 0 $B --quad cell --K 1 --N $N; done
echo SW1_LEGO_DONE
export EX_ROOT=fem_bs_ficus
for N in 384 512 640 768 1024; do $R s1_p_n$N 0 $B --N $N $PHYS; done
for N in 512 768 1024; do $R s1_c8_n$N 0 $B --quad cell --K 8 --N $N $PHYS; done
for N in 512 768 1024; do $R s1_c1_n$N 0 $B --quad cell --K 1 --N $N $PHYS; done
for d in $W/lbours/fem_bs_lego/s1_* $W/lbours/fem_bs_ficus/s1_*; do echo "== $(basename $d) $(tail -1 $d/err_fem.txt | cut -c1-200)"; done
echo SW1_DONE
