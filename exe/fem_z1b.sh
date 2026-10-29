#!/bin/bash
# 절 Z 1 차 보충: 정답 F 비율 분포 (F 가 half 로 저장돼 있어 double 로 고침) + 검산 다시 (임시 텐서 주소 버그 고침). GPU 4·5.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
OL=$W/lbours/fem_z_lego; OF=$W/lbours/fem_z_ficus
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
( CUDA_VISIBLE_DEVICES=4 python -u exe/fem_z_gtsr.py --data $W/one/L000/L000.pt --frames 80 --out $OL/gt_sr.json > $OL/gt_sr.log 2>&1
  CUDA_VISIBLE_DEVICES=4 python -u exe/fem_z_gtsr.py --data $FD --frames 125 --out $OF/gt_sr.json > $OF/gt_sr.log 2>&1 ) < /dev/null &
( for cc in 1.5 2 3 5; do CUDA_VISIBLE_DEVICES=5 python -u exe/fem_z_check.py --c $cc --T 1000; done > $OL/check.log 2>&1 ) < /dev/null &
wait
tail -n 2 $OL/gt_sr.log; tail -n 2 $OF/gt_sr.log; grep -a "\[클램프\|상대 차" $OL/check.log
rclone copy $OL r2:storage/result/anchorflow/lbours/fem_z_lego -q; rclone copy $OF r2:storage/result/anchorflow/lbours/fem_z_ficus -q
echo Z1B_DONE $(date +%T)
