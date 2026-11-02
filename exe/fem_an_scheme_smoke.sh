#!/bin/bash
# 절 AN 단계 일반화 스모크 (비측정, 10 프레임, GPU0 제외 카드 하나): 기본 (θ ½ = 옛 중점) 결과 불변, θ 1, TR-BDF2, 질량 하한, 접촉 적응 이 도는지.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
R=/tmp/dkta/lbours/fem_an/scm1; M=/tmp/dkta/lbours/fem_an/marks; mkdir -p $R $M
G=$(for i in $(seq 5); do nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits; sleep 1; done | awk -F', ' '$1>0{s[$1]+=$2} END{for (k in s) print s[k], k}' | sort -n | head -1 | awk '{print $2}')
B="--data /home/dkta/work/one/L000/L000.pt --tag L000 --frames 10 --cell 1 --newton 2 --cg 5"
for e in "base:" "th1:--scheme+theta+--theta+1" "trbdf2:--scheme+trbdf2" "mfl:--mfloor+0.25" "kc:--K_contact+2" "mgbase:--mg_levels+5+--cell+1.5"; do
  v=${e%%:*}; A=$(echo ${e#*:} | tr '+' ' '); O=$R/$v; mkdir -p $O
  CUDA_VISIBLE_DEVICES=$G python -u exe/fem_an_rt_sim.py --out $O $B $A < /dev/null > $O/log.txt 2>&1
  echo "$v $(grep -a '롤아웃\|rror' $O/log.txt | tail -1 | cut -c1-170) K $(grep -o '"K": [0-9]*' $O/fem.jsonl | sort | uniq -c | tr '\n' ' ')"
done > $M/scm1.md 2>&1
echo "카드 $G $(cat .done)" >> $M/scm1.md
rclone copy $R r2:storage/result/anchorflow/lbours/fem_an/scm1 -q; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
