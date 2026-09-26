#!/bin/bash
# 벤치용 학생 롤아웃. 같은 시나리오로 구동하고 프레임별 상태를 덤프한다.
# 사용: bench_student.sh <조합> <시드범위> <GPU>   (CK 로 체크포인트 지정)
W=${AF_WORK:-/root/work}
C=$1; SEEDS=$2; GPU=$3
export CUDA_VISIBLE_DEVICES=$GPU
export PATH=${AF_CONDA:-/opt/conda/bin}:$PATH
export AF_WORK=$W PYTHONPATH=$W/anchorflow/lib PYTHONIOENCODING=utf-8
mkdir -p $W/bench_stu
lo=${SEEDS%%-*}; hi=${SEEDS##*-}
for sd in $(seq $lo $hi); do
  S=$(printf "%02d" $sd)
  OUT=$W/bench_stu/${C}_s${S}.pt
  [ -f $OUT ] && { echo "[건너뜀] ${C}_s${S}"; continue; }
  T0=$SECONDS
  python -u $W/anchorflow/exe/roll_student.py --ck ${CK:?체크포인트를 지정하라} \
    --scen $W/bench/scen_${C}_s${S}.npz --fill $W/pgfill_${C%%_*}.npy \
    --cfg $W/bench_cfg/${C}.json --out $OUT --n_pts ${NP:-20000} \
    >> $W/bench_stu_${C}.log 2>&1
  echo "${C} s${S} stu $((SECONDS - T0)) s" >> $W/bench_stu_time.log
done
echo "BENCH_STU_DONE $C $SEEDS" >> $W/bench_progress.log
