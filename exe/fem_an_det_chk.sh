#!/bin/bash
# 절 AN 결정적 판 확인 (비측정, GPU0 제외 카드 하나): lego L000 10 프레임, 판 목록 AN_B (뉴턴:PCG:셀:MG 층, 기본 셀 1 야코비 2x5·2x20; 선탐색 10 후보 + 하강 방향) 을 결정적 두 번, 비결정 두 번 -> 끝 상태 해시·오차·프레임 ms.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
R=/tmp/dkta/lbours/fem_an/${AN_NM:-detchk}; M=/tmp/dkta/lbours/fem_an/marks; mkdir -p $R $M
G=$(for i in $(seq 5); do nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits; sleep 1; done | awk -F', ' '$1>0{s[$1]+=$2} END{for (k in s) print s[k], k}' | sort -n | head -1 | awk '{print $2}')
for b in ${AN_B:-"2:5:1:1" "2:20:1:1"}; do IFS=: read nt cg c mgl <<< "$b"; set -- $nt $cg; for m in det nd; do for r in 1 2; do
  V=c$(echo $c | tr . p)_mg${mgl}_n$1c$2_${m}_r$r; O=$R/$V; mkdir -p $O; X=""; [ $m = det ] && X="--det"
  CUDA_VISIBLE_DEVICES=$G python -u exe/fem_an_rt_sim.py --data /home/dkta/work/one/L000/L000.pt --out $O --tag L000 --frames 10 --cell $c --mg_levels $mgl --newton $1 --cg $2 --ls_n 10 --descent $X < /dev/null > $O/log.txt 2>&1
  python -c "import json; s=json.load(open('$O/summary.json')); print('$V', s.get('x_sha1'), round(100*s['err_mean_done'],4), round(s['ms_mean'] or 0, 2))" 2>&1 || grep -a rror $O/log.txt | tail -1
done; done; done > $M/${AN_NM:-detchk}.md 2>&1
echo "카드 $G $(cat .done)" >> $M/${AN_NM:-detchk}.md
rclone copy $R r2:storage/result/anchorflow/lbours/fem_an/${AN_NM:-detchk} -q; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
