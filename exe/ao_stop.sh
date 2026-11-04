#!/bin/bash
# 절 AO: 이 가지의 묶음 하나 (fem_ao_run.sh <묶음> 과 그 아래 fem_ao_sim.py) 만 끈다 -- 명령줄로 고르고, 끄기 전 ps 로 확인해 기록. 사용: bash exe/ao_stop.sh <묶음>
NM=${1:?묶음}
ps -u dkta -o pid,ppid,etime,args | grep -a "fem_ao_run.sh $NM \|fem_ao/$NM/" | grep -v grep | cut -c1-220
P=$(ps -u dkta -o pid,args | grep -a "fem_ao_run.sh $NM \|fem_ao/$NM/" | grep -v grep | awk '{print $1}')
for p in $P; do a=$(ps -o args= -p $p); case "$a" in *fem_ao_run.sh\ $NM\ *|*fem_ao/$NM/*) kill $p && echo "kill $p";; esac; done
sleep 3; echo "[남은 것]"; ps -u dkta -o pid,args | grep -a "fem_ao/$NM/\|fem_ao_run.sh $NM " | grep -v grep | cut -c1-160
