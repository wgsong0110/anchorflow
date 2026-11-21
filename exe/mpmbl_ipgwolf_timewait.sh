#!/bin/bash
# [final_fix_mpmbl] i-PG wolf N10 단독 시간 판 (50 프레임 전체): 카드에 다른 프로세스가 없어질 때까지 기다린 뒤 시작. /root/mb/ipgw_fast 가 있으면 AF_IPG_FAST=1.
REP=${1:-t}; N=${2:-10}
# 카드가 연속 5 번 (2.5 분) 비고, 기다릴 tmux 세션 (WAIT_TMUX, 쉼표) 이 모두 없어야 시작 (다른 시험 사이 틈에 끼어들지 않게)
ok=0; while [ $ok -lt 5 ]; do
  busy=0; [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && busy=1
  for t in ${WAIT_TMUX//,/ }; do tmux has-session -t $t 2>/dev/null && busy=1; done
  [ $busy = 0 ] && ok=$((ok+1)) || ok=0; sleep 30
done
[ -e /root/mb/ipgw_fast ] && export AF_IPG_FAST=1
echo "[시작] $(date -u) AF_IPG_FAST=${AF_IPG_FAST:-0}"
until bash /root/mb/code/exe/mpmbl_ipgwolf_full.sh $REP $N; do echo "[다시] $(date -u)"; sleep 30; done
