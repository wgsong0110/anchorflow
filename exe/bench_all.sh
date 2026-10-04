#!/bin/bash
# 품질기준 FPS 벤치: 1단계(탐색)는 병렬, 2단계(시간)는 **단독 실행**.
#   bash exe/bench_all.sh search     # 서브스텝 s 찾기 (GPU 0~6 병렬)
#   bash exe/bench_all.sh time       # 시간 측정 + 영상 (한 번에 하나만)
set -u
W=/home/dkta/work
PH=${1:?search 또는 time}
# 결과는 회차 디렉토리에 쌓는다 (앞 회차의 잘못된 수치와 섞이지 않게)
RUN=${AF_BENCH_RUN:-run5}
export AF_BENCH_RUN=$RUN
O=$W/bench/$RUN
SHAPES="wolf mic lego bread"
# FPS 측정은 **탄성 하나**로만 한다 (2026-10-04 지시).
MATS=${AF_BENCH_MATS:-elastic}
METHODS=${AF_BENCH_METHODS:-"pg ipg"}
# 사다리 시작점은 **CFL 위**에서 (E=2e6, rho=1e3, dx=0.02 -> s >= 370).
# 그 아래는 불안정한 궤적끼리 비교하게 되어 수렴 판정이 뜻을 잃는다.
S0_PG=${AF_S0_PG:-400}
S0_IPG=${AF_S0_IPG:-25}
MAXMUL=${AF_MAXMUL:-8}
mkdir -p $O/vid $W/wpcache
cd $W/anchorflow
# conda 초기화는 set -u 와 함께 쓰면 셸이 그 자리에서 죽는다 (미정의 변수 참조)
set +u
source /tools/anaconda3/etc/profile.d/conda.sh
conda activate af
set -u
export PYTHONPATH=$W/anchorflow/lib
export PYTHONUTF8=1

if [ "$PH" = "search" ]; then
  g=0
  for m in $METHODS; do for sh in $SHAPES; do
    S0=$S0_PG; [ "$m" = "ipg" ] && S0=$S0_IPG
    (
      for mt in $MATS; do
        CUDA_VISIBLE_DEVICES=$g python -u exe/bench_pg.py --phase search \
          --method $m --shape $sh --material $mt --frames 10 --s0 $S0 \
          --max_mul $MAXMUL --out $O/${m}_${sh}_${mt}.json
      done
    ) > $O/bs_${m}_${sh}.out 2>&1 &
    g=$(( (g + 1) % 7 ))
  done; done
  wait
  echo BENCH_SEARCH_DONE
else
  for m in $METHODS; do for sh in $SHAPES; do for mt in $MATS; do
    S0=$S0_PG; [ "$m" = "ipg" ] && S0=$S0_IPG
    CUDA_VISIBLE_DEVICES=0 python -u exe/bench_pg.py --phase time \
      --method $m --shape $sh --material $mt --frames 10 --s0 $S0 \
      --out $O/${m}_${sh}_${mt}.json \
      --vid $O/vid/${m}_${sh}_${mt}.mp4
  done; done; done
  echo BENCH_TIME_DONE
fi
