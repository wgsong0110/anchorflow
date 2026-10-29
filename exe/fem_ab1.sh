#!/bin/bash
# 절 AB 1 차: 연기 시험 (카드 1, lego N192 5 프레임 클램프 5: 기존 / 각 방식 β 0 (기존과 같아야) / β 1 (도는지)) -> 통과하면 1 차 스윕 (fem_ab_jobs.py, 카드 1·2 x 일꾼 3, 비측정) -> 표
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
OL=$W/lbours/fem_ab_lego; OF=$W/lbours/fem_ab_ficus; mkdir -p $OL $OF
OD=/tmp/dkta/lbours; mkdir -p $OD/fem_ab_lego $OD/fem_ab_ficus
# (연기 시험은 커밋 fb167cb 판에서 통과 -- lbours/fem_ab_lego/smoke.txt; 1 차 시도는 /home 가득 참으로 중단, 이 판은 노드 로컬 /tmp 에 쓴다)
for k in 0 1 2 3; do bash exe/fem_ab_gpu.sh $((1 + k % 2)) $k 4 exe/fem_ab_jobs.py > $OD/fem_ab_lego/w1_$k.log 2>&1 < /dev/null & done
wait
python exe/fem_ab_table.py $OD/fem_ab_lego/ab_l_* > $OD/fem_ab_lego/table1.md
python exe/fem_ab_table.py $OD/fem_ab_ficus/ab_f_* > $OD/fem_ab_ficus/table1.md
cat $OD/fem_ab_lego/table1.md $OD/fem_ab_ficus/table1.md
rclone copy $OD/fem_ab_lego r2:storage/result/anchorflow/lbours/fem_ab_lego -q
rclone copy $OD/fem_ab_ficus r2:storage/result/anchorflow/lbours/fem_ab_ficus -q
echo AB1_DONE $(date +%T)
