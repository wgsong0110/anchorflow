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
# 판마다 그 시점의 빈 카드 하나 (계산 프로세스 없음·메모리 < 2 GB·GPU0 제외·AO_CARDS 안·이 묶음이 이미 잡은 카드 제외), 없으면 30 초씩 기다림
free_card() {
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader > $R/smi.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader > $R/apps.txt
  awk -F', ' 'FILENAME ~ /apps/ {busy[$1]=1; next} {m=$3; sub(/ MiB/, "", m); if (!($2 in busy) && $1 != 0 && m + 0 < 2000) print $1}' $R/apps.txt $R/smi.txt | while read g; do
    ok=1; if [ -n "${AO_CARDS:-}" ]; then ok=0; for c in $AO_CARDS; do [ $g = $c ] && ok=1; done; fi
    [ -f $R/.hold_$g ] && ok=0
    [ $ok = 1 ] && { echo $g; break; }
  done
}
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $R/smi0.txt; cat $R/smi0.txt
rm -f $R/.hold_*
for e in "$@"; do
  G=""; for w in $(seq 0 ${AO_WAIT:-240}); do G=$(free_card); [ -n "$G" ] && break; sleep 30; done
  [ -z "$G" ] && { echo "[오류] 빈 카드 없음 -- ${e%%:*} 안 돌림"; continue; }
  touch $R/.hold_$G
  ( V=${e%%:*}; A=${e#*:}; O=$R/$V; mkdir -p $O
    echo "카드 $G $(cat .done) $(date '+%F %T') 인자 $A" > $O/run_info.txt
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader > $O/apps_at_start.txt
    UU=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$G '$1==g{print $2}')
    CUDA_VISIBLE_DEVICES=$G $PY -u exe/fem_ao_sim.py --data $DATA --out $O --tag wolf --frames ${AO_FR:-10} --floor vel --pbc --dump $A < /dev/null > $O/log.txt 2>&1 &
    PP=$!
    while kill -0 $PP 2>/dev/null; do nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=$UU -v t=$(date +%s) '$1==u{print t, $2}' >> $O/apps_during.txt; sleep 5; done
    wait $PP; RC=$?
    NF=$(awk -v p=$PP '$2!=p{print $2}' $O/apps_during.txt | sort -u | wc -l)
    echo "rc=$RC $(date '+%F %T') 실행 중 이 카드의 다른 계산 프로세스 $NF 개 ($(awk -v p=$PP '$2!=p{print $2}' $O/apps_during.txt | sort -u | tr '\n' ' '))" >> $O/run_info.txt
    echo "$V (카드 $G) $(grep -a '롤아웃\|rror\|대조\|발산' $O/log.txt | tail -3 | cut -c1-260 | tr '\n' ' ')"
    rclone copy $O $RR/$NM/$V -q
    rm -f $R/.hold_$G ) &
  sleep 20
done
wait
echo "[묶음 끝 $NM] $(date +%T)"; df -h /tmp | tail -1
