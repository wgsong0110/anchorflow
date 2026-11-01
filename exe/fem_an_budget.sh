#!/bin/bash
# 절 AN 반복 예산 시험 (비측정): lego L000 앞 10 프레임, MG-PCG, 뉴턴 (1·2) x PCG (5·10·20, 조기 종료 없음) x 걸음 (K 1·2), 셀 AN_CELL (1). 걸음 수마다 한 카드에서 순서대로.
# 사용: bash exe/fem_an_budget.sh <이름> [추가 인자]
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
NM=${1:-budget}; shift; EXTRA="$*"
R=/tmp/dkta/lbours/fem_an/$NM; M=/tmp/dkta/lbours/fem_an/marks; mkdir -p $R $M
C=${AN_CELL:-1}
GS=($(for i in $(seq 10); do nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits; sleep 1; done | awk -F', ' '$1>0{s[$1]+=$2} END{for (k in s) print s[k], k}' | sort -n | head -2 | awk '{print $2}'))
echo "카드 ${GS[*]} $(cat .done) $(date +%T)" > $R/info.txt
j=0
for k in 1 2; do
  G=${GS[$j]}; j=$((j + 1))
  ( for nt in 1 2; do for cg in 5 10 20; do
      V=K${k}_n${nt}_cg$cg; O=$R/$V; mkdir -p $O
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > $O/smi.csv 2>/dev/null & SM=$!
      echo "카드 $G" > $O/card.txt
      CUDA_VISIBLE_DEVICES=$G python -u exe/fem_an_sim.py --data /home/dkta/work/one/L000/L000.pt --out $O --tag L000 --floor vel --frames 10 \
        --cell $C --pc mg --K $k --integ mid --newton $nt --cg $cg --cg_tol 0 --nt_tol 0 --sec --gfin $EXTRA < /dev/null > $O/log.txt 2>&1
      kill $SM; echo "$V $(grep -a '롤아웃\|rror' $O/log.txt | tail -1 | cut -c1-160)"
      rclone copy $O r2:storage/result/anchorflow/lbours/fem_an/$NM/$V -q
    done; done ) &
done > $R/budget.out 2>&1
wait
python exe/fem_an_bench_table.py $R > $R/table.md 2>&1
cp $R/table.md $M/${NM}_table.md; cp $R/budget.out $M/${NM}.out
rclone copy $R r2:storage/result/anchorflow/lbours/fem_an/$NM -q; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
