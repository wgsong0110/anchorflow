#!/bin/bash
# 절 AN 결정적·비결정 노드별 대조 (비측정, GPU0 제외 카드 하나): lego L000 정답 상태 프레임 0·20·40, 셀 1 MG 2x5 와 셀 1 야코비 2x5.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
M=/tmp/dkta/lbours/fem_an/marks; mkdir -p $M
G=$(for i in $(seq 5); do nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits; sleep 1; done | awk -F', ' '$1>0{s[$1]+=$2} END{for (k in s) print s[k], k}' | sort -n | head -1 | awk '{print $2}')
{ echo "카드 $G $(cat .done)"
  CUDA_VISIBLE_DEVICES=$G python -u exe/fem_an_dettest.py /home/dkta/work/one/L000/L000.pt 1 5 2 5 0,20,40 2>&1 | grep -v Warning
  CUDA_VISIBLE_DEVICES=$G python -u exe/fem_an_dettest.py /home/dkta/work/one/L000/L000.pt 1 1 2 5 0,20,40 2>&1 | grep -v Warning; } > $M/dettest.md
rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
