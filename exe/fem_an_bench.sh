#!/bin/bash
# 절 AN 속도 (구간별): lego L000 앞 AN_FR(10) 프레임, 전처리 (jacobi·mg) x 걸음 (K 1·2) x 셀 (1·1.5·2 배), 한 카드에서 순서대로.
# 비측정 (기본): 카드 1 에서 6 중 10 초 평균 사용률이 가장 낮은 카드, 실행 동안 1 초마다 nvidia-smi 사용률·메모리 기록 (모든 카드).
# 단독 (AN_EXCL=1, 다음 잡 맨 앞): gpu0.lock 을 잡고 카드 0 이 빈 뒤 카드 0 에서. 사용: bash exe/fem_an_bench.sh <이름> [추가 인자]
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
NM=${1:-bench}; shift; EXTRA="$*"
R=/tmp/dkta/lbours/fem_an/$NM; M=/tmp/dkta/lbours/fem_an/marks; mkdir -p $R $M
FR=${AN_FR:-10}
CELLS=${AN_CELLS:-"1 1.5 2"}; PCS=${AN_PCS:-"jacobi mg"}; KS=${AN_KS:-"1 2"}
run_all() {
  G=$1
  echo "카드 $G $(git -C /home/dkta/work/anchorflow log -1 --format=%h 2>/dev/null) $(cat .done 2>/dev/null) $(date +%T)" > $R/info.txt
  for c in $CELLS; do for pc in $PCS; do for k in $KS; do
    V=c$(echo $c | tr . p)_${pc}_K$k; O=$R/$V; mkdir -p $O
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > $O/smi.csv 2>/dev/null & SM=$!
    CUDA_VISIBLE_DEVICES=$G python -u exe/fem_an_sim.py --data /home/dkta/work/one/L000/L000.pt --out $O --tag L000 --floor vel --frames $FR \
      --cell $c --pc $pc --K $k --integ mid --sec $EXTRA < /dev/null > $O/log.txt 2>&1
    kill $SM
    echo "$V $(grep -a '롤아웃\|rror' $O/log.txt | tail -1 | cut -c1-160)"
    rclone copy $O r2:storage/result/anchorflow/lbours/fem_an/$NM/$V -q
  done; done; done
}
if [ -n "${AN_EXCL:-}" ]; then
  ( exec 9>/home/dkta/work/gpu0.lock; flock 9
    while [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 500 ]; do sleep 10; done
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $R/gpu_at_start.txt
    run_all 0 ) > $R/bench.out 2>&1
else
  G=$(for i in $(seq 10); do nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits; sleep 1; done | awk -F', ' '$1>0{s[$1]+=$2} END{b=-1; for (k in s) if (b<0 || s[k]<s[b]) b=k; print b}')
  run_all $G > $R/bench.out 2>&1
fi
python exe/fem_an_bench_table.py $R > $R/table.md 2>&1
cp $R/table.md $M/${NM}_table.md; cp $R/bench.out $M/${NM}.out
rclone copy $R r2:storage/result/anchorflow/lbours/fem_an/$NM -q; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
