#!/usr/bin/env bash
# vast 인스턴스용 궤적 생성 워커.
#
# 인스턴스 디스크는 언제 사라질지 모르고 60GB 로는 궤적 240 개(38MB x 240 = 9GB)
# 밖에 못 담는다. 그래서 **한 묶음이 끝날 때마다 바로 R2 로 올리고 로컬은 지운다.**
# 큐는 R2 에 둔 파일을 원자적으로 집어오는 대신, 인스턴스마다 시드 구간을 미리
# 나눠 받는다 (락을 R2 로 구현하는 것보다 단순하고 실패에 강하다).
#
#   bash vast_traj_worker.sh <조합> <시작시드> <개수> [묶음크기]
set -uo pipefail
C=$1; BASE=$2; N=$3; CH=${4:-8}
W=/workspace
R2OUT="r2:storage/result/anchorflow/traj_pg2"
SH=${C%%_*}
export PYTHONPATH=$W/PG_pgtraj:$W/PG_pgtraj/gaussian-splatting:$W/anchorflow/lib
export PYTHONIOENCODING=utf-8
export AF_FLOOR_AUTO=1 AF_HANDLE=1 AF_H_D=0.25 AF_H_N=4 AF_H_MODE=randpt \
       AF_H_NONORM=1 AF_H_KIN=1 AF_H_ROUNDS=1 AF_H_RF=60 AF_H_ON=60 \
       AF_H_VMAX=0.25 AF_H_AMAX=1.0 AF_H_R=0.15
export AF_PGFILL_NPY=$W/assets/pgfill_${SH}.npy

i=0
while [ "$i" -lt "$N" ]; do
  P=""; k=0
  while [ "$k" -lt "$CH" ] && [ "$i" -lt "$N" ]; do
    P="$P${C}_t:$((BASE + i)),"; i=$((i + 1)); k=$((k + 1))
  done
  P=${P%,}
  echo "[$(date)] $C 묶음 $P" >&2
  python -u $W/anchorflow/exe/gen_handle_trajs.py \
    --pg $W/PG_pgtraj --model $W/pgmodel/x \
    --config $W/assets/wmats/${C}_t.json \
    --work $W/pgwork --out $W/out --tag ${C}_t \
    --pairs "$P" --n_pts 20000 --fill_cache $W/assets/fill_${SH}.npy
  # 생성되는 족족 올리고 로컬은 비운다
  if ls $W/out/*.pt >/dev/null 2>&1; then
    rclone copy $W/out "$R2OUT/" --transfers 4 --s3-chunk-size 64M \
      && rm -f $W/out/*.pt \
      && echo "[$(date)] 업로드 완료, 로컬 비움" >&2 \
      || echo "[$(date)] 업로드 실패 -- 로컬 유지" >&2
  fi
  rm -rf $W/pgwork/* 2>/dev/null
done
echo "VAST_WORKER_DONE $C $BASE+$N"
