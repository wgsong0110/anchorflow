#!/bin/bash
# 벤치용 GS-Verse 포팅. 사용: bench_gsv.sh <조합> <시드범위> <GPU>
W=${AF_WORK:-/root/work}
C=$1; SEEDS=$2; GPU=$3
export CUDA_VISIBLE_DEVICES=$GPU
export PATH=${AF_CONDA:-/opt/conda/bin}:$PATH
export PYTHONPATH=$W/anchorflow/lib PYTHONIOENCODING=utf-8
mkdir -p $W/bench_gsv
lo=${SEEDS%%-*}; hi=${SEEDS##*-}
for sd in $(seq $lo $hi); do
  S=$(printf "%02d" $sd)
  OUT=$W/bench_gsv/${C}_s${S}.pt
  [ -f $OUT ] && continue
  python -u $W/anchorflow/exe/bench_gsverse.py \
    --scen $W/bench/scen_${C}_s${S}.npz --fill $W/pgfill_${C%%_*}.npy \
    --cfg $W/bench_cfg/${C}.json --out $OUT >> $W/bench_gsv_${C}.log 2>&1
done
echo "BENCH_GSV_DONE $C $SEEDS" >> $W/bench_progress.log
