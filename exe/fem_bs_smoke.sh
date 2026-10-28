#!/bin/bash
# 절 M 2 차 B-스플라인 명시적 판 연기 시험 (3 프레임): 컴파일·실행·대략 속도. 잡 안에서 exe/fem_bs_boot.sh 가 부른다.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; R="bash exe/fem_ex_run.sh"
PHYS="--data $W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --tag F_pg_v0 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
export EX_ROOT=fem_bs_lego
$R smk_p_opt_n40 0 --interp bspline2 --mode sub --N 40 --opt --frames 4 --no_dump
$R smk_p_base_n40 0 --interp bspline2 --mode sub --N 40 --frames 4 --no_dump
$R smk_c8_opt_n40 0 --interp bspline2 --quad cell --K 8 --mode sub --N 40 --opt --frames 4 --no_dump
$R smk_p_nog_n40 0 --interp bspline2 --mode sub --N 40 --opt --no_graph --frames 4 --no_dump
export EX_ROOT=fem_bs_ficus
$R smk_p_opt_n300 0 --interp bspline2 --mode sub --N 300 --opt --frames 4 --no_dump $PHYS
$R smk_c8_opt_n300 0 --interp bspline2 --quad cell --K 8 --mode sub --N 300 --opt --frames 4 --no_dump $PHYS
for d in $W/lbours/fem_bs_lego/smk_* $W/lbours/fem_bs_ficus/smk_*; do echo "== $d"; cat $d/EX_FAIL 2>/dev/null; grep -a "CFL\|프레임 [0-9]/\|롤아웃\]\|발산\|Error\|error" $d/v_fem.log | cut -c1-260 | tail -8; done
echo SMOKE_DONE
