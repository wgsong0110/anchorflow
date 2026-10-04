#!/bin/bash
# 품질기준 FPS 벤치: 1단계(탐색)는 병렬, 2단계(시간)는 **단독 실행**.
#   bash exe/bench_all.sh search     # 서브스텝 s 찾기 (GPU 0~6 병렬)
#   bash exe/bench_all.sh time       # 시간 측정 + 영상 (한 번에 하나만)
set -u
W=/home/dkta/work
PH=${1:?search 또는 time}
SHAPES="wolf mic lego bread"
MATS="elastic elastoplastic viscoplastic fracture"
mkdir -p $W/bench/vid $W/wpcache
cd $W/anchorflow
source /tools/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate af 2>/dev/null || true
export PYTHONPATH=$W/anchorflow/lib

if [ "$PH" = "search" ]; then
  g=0
  for m in pg ipg; do for sh in $SHAPES; do
    (
      for mt in $MATS; do
        CUDA_VISIBLE_DEVICES=$g python -u exe/bench_pg.py --phase search \
          --method $m --shape $sh --material $mt --frames 10 --s0 50 \
          --out $W/bench/${m}_${sh}_${mt}.json
      done
    ) > $W/bs_${m}_${sh}.out 2>&1 &
    g=$(( (g + 1) % 7 ))
  done; done
  wait
  echo BENCH_SEARCH_DONE
else
  for m in pg ipg; do for sh in $SHAPES; do for mt in $MATS; do
    CUDA_VISIBLE_DEVICES=0 python -u exe/bench_pg.py --phase time \
      --method $m --shape $sh --material $mt --frames 10 --s0 50 \
      --out $W/bench/${m}_${sh}_${mt}.json \
      --vid $W/bench/vid/${m}_${sh}_${mt}.mp4
  done; done; done
  echo BENCH_TIME_DONE
fi
