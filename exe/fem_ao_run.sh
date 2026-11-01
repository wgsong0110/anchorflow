#!/bin/bash
# 절 AO: wolf 암시적 B-스플라인 모래 판 실행 묶음 (잡 안). 판마다 빈 카드 하나 (계산 프로세스 없음, GPU0 제외; AO_CARDS 로 후보 제한), 카드가 모자라면 같은 카드에서 순서대로.
# 사용: AO_FR=10 bash exe/fem_ao_run.sh <묶음> "<판>:<인자>" ...   출력 /tmp/dkta/lbours/fem_ao/<묶음>/<판> -> R2 lbours/fem_ao/<묶음>/<판> (끝나면 바로)
cd "$(dirname "$0")/.." || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8 PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH
export CUPY_CACHE_DIR=/tmp/dkta/cache/cupy XDG_CACHE_HOME=/tmp/dkta/cache TORCH_EXTENSIONS_DIR=/tmp/dkta/cache/torch_ext
RR=r2:storage/result/anchorflow/lbours/fem_ao
NM=$1; shift
R=/tmp/dkta/lbours/fem_ao/$NM; mkdir -p $R /tmp/dkta/ag_data $CUPY_CACHE_DIR
DATA=/tmp/dkta/ag_data/wolf.pt
[ -s $DATA ] || rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wolf.pt $DATA -q
echo "[묶음 $NM] $(cat .done) $(date +%T) 데이터 $(ls -la $DATA | awk '{print $5}')"; df -h /tmp | tail -1
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader > $R/smi0.txt
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $R/apps0.txt
GS=($(awk -F', ' 'FILENAME ~ /apps0/ {busy[$1]=1; next} {m=$3; sub(/ MiB/, "", m); if (!($2 in busy) && $1 != 0 && m + 0 < 2000) print $1}' $R/apps0.txt $R/smi0.txt))
if [ -n "${AO_CARDS:-}" ]; then GS=($(for g in ${GS[@]}; do for c in $AO_CARDS; do [ $g = $c ] && echo $g; done; done)); fi
echo "빈 카드 ${GS[*]}"; cat $R/smi0.txt
[ ${#GS[@]} -eq 0 ] && { echo "[오류] 빈 카드 없음"; exit 1; }
NV=$#; VL=("$@")
for j in $(seq 0 $((${#GS[@]} - 1))); do
  ( G=${GS[$j]}
    for i in $(seq 0 $((NV - 1))); do
      [ $((i % ${#GS[@]})) -eq $j ] || continue
      e=${VL[$i]}; V=${e%%:*}; A=${e#*:}; O=$R/$V; mkdir -p $O
      echo "카드 $G $(cat .done) $(date '+%F %T') 인자 $A" > $O/run_info.txt
      CUDA_VISIBLE_DEVICES=$G $PY -u exe/fem_ao_sim.py --data $DATA --out $O --tag wolf --frames ${AO_FR:-10} --floor vel --pbc --dump $A < /dev/null > $O/log.txt 2>&1
      echo "rc=$?" >> $O/run_info.txt
      echo "$V (카드 $G) $(grep -a '롤아웃\|rror\|대조\|발산' $O/log.txt | tail -3 | cut -c1-260 | tr '\n' ' ')"
      rclone copy $O $RR/$NM/$V -q
    done ) &
  sleep 3
done
wait
echo "[묶음 끝 $NM] $(date +%T)"; df -h /tmp | tail -1
