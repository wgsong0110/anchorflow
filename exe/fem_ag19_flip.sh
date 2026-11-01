#!/bin/bash
# 절 AG.19 (사용자 지시): 정답 사면체 안 입자 순서 뒤집힘 직접 측정 -- 계산 노드 잡 컨테이너 CPU (GPU 안 씀)
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=""; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
D=/tmp/dkta/flip; mkdir -p $D
python -I exe/fem_ag19_flip.py /tmp/dkta/ag_data/wm.pt 0.0094281 1,2,3,4,5,6 0.02 $D/wm.json > $D/wm.log 2>&1 &
python -I exe/fem_ag19_flip.py /home/dkta/work/one/L000/L000.pt 0.028284 all 0.0 $D/L000.json > $D/L000.log 2>&1 &
python -I exe/fem_ag19_flip.py /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt 0.056569 all none $D/ficus.json > $D/ficus.log 2>&1 &
wait
rclone copy $D r2:storage/result/anchorflow/lbours/fem_ag_wm/flip -q
echo FLIP_DONE
