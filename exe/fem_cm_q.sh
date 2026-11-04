#!/bin/bash
# 절 CM 단독 시간 목록 (카드 0, gpu0.lock = 절 AN·AO 단독 측정과 같은 잠금; 판마다 카드 0 의 다른 계산 프로세스 0 일 때 시작, 실행 동안 5 초마다 기록):
# bash exe/fem_cm_q.sh <묶음> <목록> -- 줄 "<이름> <씬> <공통 인자...>". 비결정 판, 시간 = 재생 GPU 시간 (첫 2 프레임 뺌).
cd "$(dirname "$0")/.." || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8 PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_cmq XDG_CACHE_HOME=/dev/shm/dkta/cache
NM=$1; LST=$2; R=/dev/shm/dkta/lbours/fem_cm/$NM; mkdir -p $R $CUPY_CACHE_DIR; RR=r2:storage/result/anchorflow/lbours/fem_cm/$NM
source exe/fem_cm_scn.sh
U0=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
others0() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 9>&- | grep -c "$U0"; }
while read -r v t A; do [ -z "$v" ] && continue; [ "${v:0:1}" = "#" ] && continue; cm_data $t; done < $LST          # 자료 먼저 (잠금 밖)
exec 9>/home/dkta/work/gpu0.lock; flock 9
echo "잠금 $(date +%T) $(cat .done)" | tee -a $R/run.out
while read -r v t A; do [ -z "$v" ] && continue; [ "${v:0:1}" = "#" ] && continue
  O=$R/$v; mkdir -p $O
  cm_scene $t $A
  while [ "$(others0)" -gt 0 ]; do sleep 5; done
  echo "시작 때 카드 0 의 다른 프로세스 $(others0) 개 $(date +%T) $SIM $DA $XA" > $O/card.txt
  CUDA_VISIBLE_DEVICES=0 timeout ${CM_TMO:-1500} $PY -u $SIM --out $O --tag $t $DA $XA 9>&- < /dev/null > $O/log.txt 2>&1 &
  PP=$!
  while kill -0 $PP 2>/dev/null; do nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 9>&- | awk -F', ' -v u=$U0 -v t=$(date +%s) '$1==u{print t, $2}' >> $O/apps_during.txt; sleep 5; done
  wait $PP; RC=$?
  NF=$(( $(awk '{print $2}' $O/apps_during.txt 2>/dev/null | sort -u | wc -l) - 1 )); [ $NF -lt 0 ] && NF=0              # 기록 pid 하나는 이 실행 자신 (timeout 의 자식)
  echo "rc=$RC 끝 $(date +%T) 실행 중 카드 0 의 다른 계산 프로세스 $NF 개" >> $O/card.txt
  echo "$v $t | $(grep -a '롤아웃\|rror\|발산' $O/log.txt | tail -2 | cut -c1-200 | tr '\n' ' ') | 다른 $NF" | tee -a $R/run.out
  rclone copy $O $RR/$v -q --exclude pred.pt
  rclone copy $R/run.out $RR -q
done < $LST
flock -u 9; exec 9>&-
echo "[끝 $NM] $(date '+%F %T')" | tee -a $R/run.out; rclone copy $R/run.out $RR -q
