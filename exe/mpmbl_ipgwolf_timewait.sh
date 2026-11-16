#!/bin/bash
# [final_fix_mpmbl] i-PG wolf N10 단독 시간 판 (50 프레임 전체): 카드에 다른 프로세스가 없어질 때까지 기다린 뒤 시작. /root/mb/ipgw_fast 가 있으면 AF_IPG_FAST=1.
REP=${1:-t}; N=${2:-10}
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
[ -e /root/mb/ipgw_fast ] && export AF_IPG_FAST=1
echo "[시작] $(date -u) AF_IPG_FAST=${AF_IPG_FAST:-0}"
until bash /root/mb/code/exe/mpmbl_ipgwolf_full.sh $REP $N; do echo "[다시] $(date -u)"; sleep 30; done
