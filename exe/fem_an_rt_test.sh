#!/bin/bash
# 절 AN 실시간 판 시험 (비측정, 카드 AN_G): lego L000 10 프레임 셀 1·1.5·2 x (뉴턴 1 PCG 5, 뉴턴 2 PCG 5, 뉴턴 2 PCG 10) + ficus 10 프레임 (구속 확인). torch 가속 판과 오차 비교용.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
NM=${1:-rt1}; R=/tmp/dkta/lbours/fem_an/$NM; M=/tmp/dkta/lbours/fem_an/marks; mkdir -p $R $M
G=${AN_G:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1>0{print $2, $1}' | sort -n | head -1 | awk '{print $2}')}
echo "카드 $G $(cat .done) $(date +%T)" > $R/info.txt
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> $R/info.txt
for c in 1 1.5 2; do for b in "1 5" "2 5" "2 10"; do set -- $b
  V=c$(echo $c | tr . p)_n$1_cg$2; O=$R/$V; mkdir -p $O
  CUDA_VISIBLE_DEVICES=$G python -u exe/fem_an_rt_sim.py --data /home/dkta/work/one/L000/L000.pt --out $O --tag L000 --frames 10 --cell $c --newton $1 --cg $2 < /dev/null > $O/log.txt 2>&1
  echo "$V $(grep -a '롤아웃\|rror' $O/log.txt | tail -1 | cut -c1-170)"
done; done > $R/run.out 2>&1
O=$R/ficus; mkdir -p $O
CUDA_VISIBLE_DEVICES=$G python -u exe/fem_an_rt_sim.py --data /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --out $O --tag ficus --frames 10 --floor none --cfg_bc --region_density --pg_vdamp \
  --newton 2 --cg 10 < /dev/null > $O/log.txt 2>&1
echo "ficus $(grep -a '롤아웃\|rror' $O/log.txt | tail -1 | cut -c1-170)" >> $R/run.out
cp $R/run.out $M/${NM}.out; for v in $(ls $R); do [ -f $R/$v/log.txt ] && cp $R/$v/log.txt $M/${NM}_$v.log; done
rclone copy $R r2:storage/result/anchorflow/lbours/fem_an/$NM -q; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
