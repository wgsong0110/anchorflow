#!/bin/bash
# 절 AO 단독 측정 (잡 안): GPU0 을 gpu0.lock (절 AN 단독 측정과 같은 잠금) 으로 잡고, 판마다 카드 0 의 다른 계산 프로세스가 0 일 때 시작, 실행 동안 5 초마다 기록.
# 사용: AO_FR=50 AO_SIM=fem_ao_rt_sim.py bash exe/fem_ao_excl.sh <묶음> "<판>:<인자>" ...
cd "$(dirname "$0")/.." || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8 PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH
export CUPY_CACHE_DIR=/tmp/dkta/cache/cupy XDG_CACHE_HOME=/tmp/dkta/cache TORCH_EXTENSIONS_DIR=/tmp/dkta/cache/torch_ext
NM=$1; shift; R=/tmp/dkta/lbours/fem_ao/$NM; RR=r2:storage/result/anchorflow/lbours/fem_ao/$NM; mkdir -p $R
SCN=${AO_SCN:-wolf}; DATA=/tmp/dkta/ag_data/$SCN.pt
U0=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
others0() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$U0"; }
exec 9>/home/dkta/work/gpu0.lock; flock 9
echo "[잠금] $(date '+%F %T')"
for e in "$@"; do
  V=${e%%:*}; A=${e#*:}; O=$R/$V; mkdir -p $O
  while [ "$(others0)" -gt 0 ]; do sleep 5; done
  echo "카드 0 $(cat .done) $(date '+%F %T') 시작 때 다른 프로세스 $(others0) 인자 $A" > $O/run_info.txt
  CUDA_VISIBLE_DEVICES=0 $PY -u exe/${AO_SIM:-fem_ao_rt_sim.py} --data $DATA --out $O --tag $SCN --frames ${AO_FR:-50} $A < /dev/null > $O/log.txt 2>&1 9>&- &
  PP=$!
  while kill -0 $PP 2>/dev/null; do nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 9>&- | awk -F', ' -v u=$U0 -v t=$(date +%s) '$1==u{print t, $2}' >> $O/apps_during.txt; sleep 5; done
  wait $PP; RC=$?
  NF=$(awk -v p=$PP '$2!=p{print $2}' $O/apps_during.txt | sort -u | wc -l)
  echo "rc=$RC $(date '+%F %T') 실행 중 카드 0 의 다른 계산 프로세스 $NF 개" >> $O/run_info.txt
  echo "$V 다른 $NF $(grep -a '롤아웃\|rror' $O/log.txt | tail -1 | cut -c1-200)"
  rclone copy $O $RR/$V -q
done
exec 9>&-
