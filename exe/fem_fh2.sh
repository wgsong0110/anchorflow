#!/bin/bash
# 절 Y 통계 다시 (fem_fh_stats.py 수정판: 셀 갈아탐 = 셀 동료 집합 변화, 입자별 희소 지속) -- fem_fh.sh 의 실행 4 개가 끝나길 (FH_DONE) 기다려 GPU 0~3 에서.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
O=$W/lbours/fem_fh_lego
until [ -f $O/FH_DONE ] || [ -f $O/FH_SMOKE_FAIL ]; do sleep 60; done
for g in 0 1 2 3; do
  d=$O/fh_tet_n192_r$g
  ( CUDA_VISIBLE_DEVICES=$g python -u exe/fem_fh_stats.py $d < /dev/null > $d/fh_stats2.log 2>&1; echo "rc $?" >> $d/fh_stats2.log ) &
done
wait
for g in 0 1 2 3; do echo "== r$g"; grep -a "발산 프레임\|폭주 셀\|완료\|갈아탐\|rc \|Error" $O/fh_tet_n192_r$g/fh_stats2.log | cut -c1-600; done
rclone copy $O r2:storage/result/anchorflow/lbours/fem_fh_lego -q --exclude "fhist/**"
rclone copy $O r2:storage/result/anchorflow/lbours/fem_fh_lego -q
touch $O/FH2_DONE
echo FH2_DONE $(date +%T)
