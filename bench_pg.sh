#!/bin/bash
# 벤치용 PG 기준 궤적. 시나리오 파일로 구동하고 프레임별 상태를 덤프한다.
# 사용: bench_pg.sh <조합> <시드범위> <GPU>   (예: bench_pg.sh mic_clayC 0-1 3)
W=${AF_WORK:-/home/dkta/work}
C=$1; SEEDS=$2; GPU=$3
SH_=${C%%_*}
export CUDA_VISIBLE_DEVICES=$GPU
export PATH=${AF_CONDA:-/home/dkta/.conda/envs/af/bin}:$PATH
# 인스턴스에서는 AF_CONDA=/opt/conda/bin, AF_WORK=/root/work 로 준다
export PYTHONPATH=$W/anchorflow/lib PYTHONIOENCODING=utf-8 PYTHONUTF8=1
export LANG=C.UTF-8 LC_ALL=C.UTF-8
export AF_FLOOR_AUTO=1 AF_HANDLE=1 AF_H_KIN=1 AF_H_N=2 AF_H_NONORM=1 \
       AF_H_R=0.15 AF_H_ROUNDS=1 AF_H_RF=240 AF_H_ON=240 AF_H_MODE=randpt
mkdir -p $W/bench_pg
lo=${SEEDS%%-*}; hi=${SEEDS##*-}
for sd in $(seq $lo $hi); do
  S=$(printf "%02d" $sd)
  [ -f $W/bench_pg/${C}_s${S}.pt ] && { echo "[건너뜀] ${C}_s${S}"; continue; }
  export AF_H_SCEN=$W/bench/scen_${C}_s${S}.npz
  T0=$SECONDS
  AF_PGFILL_NPY=$W/pgfill_${SH_}.npy python -u $W/anchorflow/exe/gen_handle_trajs.py \
    --pg $W/PG_pgtraj --model $W/pgmodel/${SH_}_whitebg-trained \
    --config $W/bench_cfg/${C}.json --work $W/wbench_${C}_$GPU \
    --out $W/bench_pg --tag ${C} --pairs ${C}:${sd} --n_pts 20000 \
    >> $W/bench_pg_${C}.log 2>&1
  DT=$((SECONDS - T0))
  echo "${C} s${S} $DT s" >> $W/bench_pg_time.log
  # 덤프에 벽시계와 FPS 를 박아 넣는다 (지표 스크립트가 여기서 읽는다)
  PYTHONPATH=$W/anchorflow/lib python - "$W/bench_pg/${C}_s${S}.pt" "$DT" <<'PY' >> $W/bench_pg_${C}.log 2>&1
import sys, torch
p, dt = sys.argv[1], float(sys.argv[2])
d = torch.load(p, map_location="cpu", weights_only=False)
n = int(d["x"].shape[0]) - 1
d["wall_s"] = dt
d["fps"] = n / max(dt, 1e-9)
torch.save(d, p)
print(f"[시간] {p} {dt:.1f}s {n / max(dt, 1e-9):.3f} FPS")
PY
done
echo "BENCH_PG_DONE $C $SEEDS" >> $W/bench_pg_progress.log
