#!/bin/bash
# 절 AN 속도·오차 표 (비측정): lego L000 앞 AN_FR(10) 프레임, 셀 (AN_CELLS) x 전처리 (AN_PCS) x 걸음 (AN_KS) x 뉴턴 (1·2) x PCG (5·10·20), 조기 종료 없음.
# 가장 한가한 카드 AN_NCARD(3) 장에 판을 돌려 가며 나눠 카드마다 순서대로. 구간별 (--sec --gfin) 한 번 + 동기화 없는 프레임 시간 (AN_NS=1 이면) 한 번.
# 사용: bash exe/fem_an_sweep.sh <이름> [추가 인자]
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
NM=${1:-sweep}; shift; EXTRA="$*"
M=/tmp/dkta/lbours/fem_an/marks; mkdir -p $M
NC=${AN_NCARD:-3}
GS=($(for i in $(seq 10); do nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits; sleep 1; done | awk -F', ' '$1>0{s[$1]+=$2} END{for (k in s) print s[k], k}' | sort -n | head -$NC | awk '{print $2}'))
VL=()
for c in ${AN_CELLS:-1 1.5 2}; do for pc in ${AN_PCS:-jacobi mg}; do for k in ${AN_KS:-1}; do for nt in 1 2; do for cg in 5 10 20; do
  VL+=("c$(echo $c | tr . p)_${pc}_K${k}_n${nt}_cg${cg}:--cell $c --pc $pc --K $k --newton $nt --cg $cg"); done; done; done; done; done
for MODE in sec ${AN_NS:+ns}; do
  R=/tmp/dkta/lbours/fem_an/${NM}_$MODE; mkdir -p $R
  echo "카드 ${GS[*]} $(cat .done) $(date +%T)" > $R/info.txt
  SF="--sec --gfin"; [ $MODE = ns ] && SF=""
  for j in $(seq 0 $((${#GS[@]} - 1))); do
    ( G=${GS[$j]}; i=0
      for e in "${VL[@]}"; do
        if [ $((i % ${#GS[@]})) -eq $j ]; then
          V=${e%%:*}; A=${e#*:}; O=$R/$V; mkdir -p $O; echo "카드 $G" > $O/card.txt
          nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > $O/smi.csv 2>/dev/null & SM=$!
          CUDA_VISIBLE_DEVICES=$G python -u exe/fem_an_sim.py --data /home/dkta/work/one/L000/L000.pt --out $O --tag L000 --floor vel --frames ${AN_FR:-10} \
            --integ mid --cg_tol 0 --nt_tol 0 $A $SF $EXTRA < /dev/null > $O/log.txt 2>&1
          kill $SM; echo "$V $(grep -a '롤아웃\|rror' $O/log.txt | tail -1 | cut -c1-140)"
          rclone copy $O r2:storage/result/anchorflow/lbours/fem_an/${NM}_$MODE/$V -q
        fi; i=$((i + 1))
      done ) &
  done > $R/run.out 2>&1
  wait
  python exe/fem_an_bench_table.py $R > $R/table.md 2>&1
  cp $R/table.md $M/${NM}_${MODE}_table.md; cp $R/run.out $M/${NM}_$MODE.out
  rclone copy $R r2:storage/result/anchorflow/lbours/fem_an/${NM}_$MODE -q; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
done
