#!/bin/bash
# 절 AN 걸음 끝 |g|/|g0| (비측정, GPU0 제외 카드 하나): lego L000 10 프레임, 판 목록 AN_LIST (셀:전처리:뉴턴:PCG), 추가 인자 AN_X. 표 = 판마다 걸음별 |g_끝|/|g_처음| 의 프레임 평균·최대.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
NM=${1:-gfin1}; R=/tmp/dkta/lbours/fem_an/$NM; M=/tmp/dkta/lbours/fem_an/marks; mkdir -p $R $M
G=$(for i in $(seq 5); do nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits; sleep 1; done | awk -F', ' '$1>0{s[$1]+=$2} END{for (k in s) print s[k], k}' | sort -n | head -1 | awk '{print $2}')
for e in $AN_LIST; do IFS=: read c pc nt cg <<< "$e"; MGL=1; [ $pc = mg ] && MGL=5
  V=c$(echo $c | tr . p)_${pc}_n${nt}_cg$cg; O=$R/$V; mkdir -p $O
  CUDA_VISIBLE_DEVICES=$G python -u exe/fem_an_rt_sim.py --data /home/dkta/work/one/L000/L000.pt --out $O --tag L000 --frames 10 --cell $c --newton $nt --cg $cg --mg_levels $MGL --gfin ${AN_X:-} < /dev/null > $O/log.txt 2>&1
  python - <<PY
import json, numpy as np
r=[json.loads(l) for l in open('$O/fem.jsonl')]
g=np.array([x['grel'] for x in r if 'grel' in x])
print('$V', '걸음 끝 |g|/|g0| 평균 %.2e 최대 %.2e' % (g.mean(), g.max()) if g.size else '없음', '10 프레임 오차 %.3f%%' % (100*np.mean([x['err'] for x in r])))
PY
done > $M/${NM}.md 2>&1
echo "카드 $G $(cat .done)" >> $M/${NM}.md
rclone copy $R r2:storage/result/anchorflow/lbours/fem_an/$NM -q; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
