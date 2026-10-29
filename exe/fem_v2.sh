#!/bin/bash
# 절 V 2 차: 안정화 β (셀 적분 혼합) 스윕 -- GPU 0~6 에 나눠 동시에 (exe/fem_v_jobs2.py, 비측정; 실행마다 카드 빈 것 확인) -> β 별 표·ficus 주기
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
python exe/fem_v_jobs2.py | wc -l
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
for G in 0 1 2 3 4 5 6; do bash exe/fem_v_gpu.sh $G exe/fem_v_jobs2.py 0 < /dev/null > $W/fem_v2_g$G.out 2>&1 & done
wait
for B in 05 10 20 30 50; do
  python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_v_ficus/v_f_b${B}_*/pred.pt > $W/lbours/fem_v_ficus/period2_b$B.txt 2>&1
  python exe/fem_r_table.py $W/lbours/fem_v_lego/v_l_b${B}_[oL]_* > $W/lbours/fem_v_lego/table2_b$B.md
  python exe/fem_r_table.py $W/lbours/fem_v_ficus/v_f_b${B}_[oL]_* > $W/lbours/fem_v_ficus/table2_b$B.md
done
rclone copy $W/lbours/fem_v_lego r2:storage/result/anchorflow/lbours/fem_v_lego -q
rclone copy $W/lbours/fem_v_ficus r2:storage/result/anchorflow/lbours/fem_v_ficus -q
echo V2_DONE $(date +%T)
