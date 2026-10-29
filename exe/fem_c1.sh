#!/bin/bash
# 절 C 1 차: 기존 판 구간별 프로파일 (단독 GPU0, 프레임 41 하나를 torch.profiler 로 -- 그래프 끔) + 그래프 판 ms (41 프레임)
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
LB="--mode lag --opt --mass tet --floor vel --rstat --no_dump --frames 41"
T="env EX_TIMED=1 EX_ROOT=fem_c_lego bash exe/fem_c_run.sh"
$T p0_c5_n256 0 $LB --lim_sr 5 --N 256 --no_graph --tprof 40 < /dev/null > /dev/null 2>&1
$T p0_c2_n192 0 $LB --lim_sr 2 --N 192 --no_graph --tprof 40 < /dev/null > /dev/null 2>&1
$T p0_d03_n192 0 $LB --lim 0.3 --lim_mode box --N 192 --no_graph --tprof 40 < /dev/null > /dev/null 2>&1
$T p0_none_n192 0 $LB --N 192 --no_graph --tprof 10 --frames 11 < /dev/null > /dev/null 2>&1
echo C1_DONE $(date +%T)
