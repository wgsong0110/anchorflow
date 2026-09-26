#!/bin/bash
# 한 (조합, 시드) 에 대해 있는 덤프 전부의 지표를 낸다. 큐 워커와 같은 규약.
W=${AF_WORK:-/root/work}
C=$1; SEEDS=$2; GPU=$3
export CUDA_VISIBLE_DEVICES=$GPU
export PATH=${AF_CONDA:-/opt/conda/bin}:$PATH
export PYTHONPATH=$W/anchorflow/lib PYTHONIOENCODING=utf-8
mkdir -p $W/bench_met
lo=${SEEDS%%-*}; hi=${SEEDS##*-}
for sd in $(seq $lo $hi); do
  S=$(printf "%02d" $sd)
  REF=$W/bench_pg/${C}_s${S}.pt
  [ -f $REF ] || { echo "[없음] PG ${C}_s${S}"; continue; }
  for kind in pg gsv stu ipg20f60 ipg8f60; do
    case $kind in
      pg)  D=$REF;;
      *)   D=$W/bench_${kind#bench_}/${C}_s${S}.pt; D=$W/bench_$kind/${C}_s${S}.pt;;
    esac
    [ -f "$D" ] || continue
    python -u $W/anchorflow/exe/bench_metrics.py --ref $REF --tgt $D \
      --cfg $W/bench_cfg/${C}.json --tag $kind --combo $C --seed $sd \
      --out $W/bench_met/metrics_${kind}_${C}_s${S}.csv \
      >> $W/bench_met_${C}.log 2>&1
  done
done
echo "BENCH_MET_DONE $C $SEEDS" >> $W/bench_progress.log
