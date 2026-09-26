#!/bin/bash
# 벤치용 i-PhysGaussian. 같은 시나리오 파일로 구동하고 h5 로 상태를 덤프한다.
# 사용: bench_ipg.sh <조합> <시드범위> <GPU>    (DTM 으로 dt 배수 지정)
W=${AF_WORK:-/root/work}
C=$1; SEEDS=$2; GPU=$3
SH_=${C%%_*}; DTM=${DTM:-20}; FR=${FR:-60}
export CUDA_VISIBLE_DEVICES=$GPU
export PATH=${AF_CONDA:-/opt/conda/bin}:$PATH
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export AF_IPG_NEWTON=1 AF_H_R=0.15
mkdir -p $W/bench_ipg${DTM}f${FR}
lo=${SEEDS%%-*}; hi=${SEEDS##*-}
for sd in $(seq $lo $hi); do
  S=$(printf "%02d" $sd)
  OUT=$W/bench_ipg${DTM}f${FR}/${C}_s${S}
  [ -f $OUT.pt ] && { echo "[건너뜀] ${C}_s${S}"; continue; }
  export AF_H_SCEN=$W/bench/scen_${C}_s${S}.npz AF_PGFILL_NPY=$W/pgfill_${SH_}.npy
  T0=$SECONDS
  rm -rf $OUT.h5dir && mkdir -p $OUT.h5dir
  (cd $W/i-physgaussian && python -u gs_simulation.py \
     --model_path $W/pgmodel/${SH_}_whitebg-trained \
     --config $W/bench_cfg_f$FR/${C}.json --output_path $OUT.h5dir \
     --output_h5 --implicit --solver newton_gmres --dt_multiplier $DTM \
     --white_bg) >> $W/bench_ipg_${C}_dt${DTM}f${FR}.log 2>&1
  echo "${C} s${S} dt$DTM f$FR $((SECONDS - T0)) s" >> $W/bench_ipg_time.log
  PYTHONPATH=$W/anchorflow/lib python -u $W/anchorflow/exe/h5_to_pt.py \
    --dir $OUT.h5dir --out $OUT.pt --cfg $W/bench_cfg_f$FR/${C}.json \
    >> $W/bench_ipg_${C}_dt${DTM}f${FR}.log 2>&1
done
echo "BENCH_IPG_DONE $C $SEEDS dt$DTM" >> $W/bench_progress.log
