#!/bin/bash
# 절 AG.14: wolf 입자 적분점 증분 RM 판 (클램프 끔) 을 절 AG.12 인과 진단 실행 (ag_wolf_n640_cx1/cx2) 과 같은 명령으로 다시 돌리며
#   입자 추적 (--akpl_track) + 끝난 프레임 위치 덤프 (--dump_pred --dump_div). 비측정, 카드 둘 (빈 카드만, GPU0 금지).
# 사용: bash exe/fem_ag14_track.sh <카드 B> <카드 A>   (B = node_h 그대로, A = node_h 2 배)
GB=${1:?카드 B}; GA=${2:?카드 A}
for g in $GB $GA; do case $g in 1|2|3|4|5|6) ;; *) echo "[오류] 카드 1~6 만"; exit 1;; esac; done
cd "$(dirname "$0")/.." || exit 1
BASE="--data /tmp/dkta/ag_data/wolf.pt --tag wolf --frames 50 --node_h 0.014142 --floor vel --pbc --plast --mode lag --opt --mass tet --rstat --lim_sr 5 --cfuse --cfuse_mode 2 --sp vcap --sp_vcap 0.5 --no_dump --N 640 --lim_sr 1e30 --sp_vcap 1e9 --ak_pq --ak_pq_plast inc --no_graph --akpl_cause --akpl_k 30 --div 1e9 --akpl_track --dump_pred --dump_div"
EX_ROOT=fem_ag_wolf bash exe/fem_ab_run.sh ag_wolf_n640_tk1_r1 $GB $BASE < /dev/null > /tmp/dkta/ag14_tk1.out 2>&1 &
EX_ROOT=fem_ag_wolf bash exe/fem_ab_run.sh ag_wolf_n640_tk2_r1 $GA $BASE --node_h 0.028284 < /dev/null > /tmp/dkta/ag14_tk2.out 2>&1 &
wait
echo "[절 AG.14 추적] 끝 $(date +%T)"
