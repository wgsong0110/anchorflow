#!/bin/bash
# 절 AN 결과 불변 확인 (비측정, GPU0 제외 카드 하나): lego L000 10 프레임, 판 목록 AN_LIST (셀:전처리:뉴턴:PCG) x 변형 AN_VARS (이름:플래그, 빈칸 +), 추가 AN_X -> 오차·프레임 ms.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
NM=${1:-inv1}; R=/tmp/dkta/lbours/fem_an/$NM; M=/tmp/dkta/lbours/fem_an/marks; mkdir -p $R $M
G=$(for i in $(seq 5); do nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits; sleep 1; done | awk -F', ' '$1>0{s[$1]+=$2} END{for (k in s) print s[k], k}' | sort -n | head -1 | awk '{print $2}')
for e in $AN_LIST; do IFS=: read c pc nt cg <<< "$e"; MGL=1; [ $pc = mg ] && MGL=5
  for vv in $AN_VARS; do vn=${vv%%:*}; VA=$(echo ${vv#*:} | tr '+' ' ')
    V=c$(echo $c | tr . p)_${pc}_n${nt}_cg${cg}_$vn; O=$R/$V; mkdir -p $O
    CUDA_VISIBLE_DEVICES=$G python -u exe/fem_an_rt_sim.py --data /home/dkta/work/one/L000/L000.pt --out $O --tag L000 --frames 10 --cell $c --newton $nt --cg $cg --mg_levels $MGL ${AN_X:-} $VA < /dev/null > $O/log.txt 2>&1
    echo "$V $(grep -a '롤아웃\|rror' $O/log.txt | tail -1 | cut -c1-170)"
  done; done > $M/${NM}.md 2>&1
echo "카드 $G $(cat .done)" >> $M/${NM}.md
rclone copy $R r2:storage/result/anchorflow/lbours/fem_an/$NM -q; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
