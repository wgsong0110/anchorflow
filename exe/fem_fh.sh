#!/bin/bash
# 사용자 10-29 "셀 집중 질량 판 (절 R 설정, 제한 끔, lego N192) 에서 폭주 노드 이웃 셀의 응력이 왜 프레임 시작부터 큰가 -- 입자 F 이력":
#   절 U (1) 설정 그대로 (--mode lag --opt --mass tet --floor vel --N 192, 제한 끔) + --lagdump 3 (폭주 노드 판정) + --fhist (프레임마다 입자 F·셀 F̄·끝 U, 계산 불변)
#   4 회 (원자 합 순서로 발산 프레임이 갈린다, 카드마다 하나, 비측정) -> exe/fem_fh_stats.py (GPU) 로 폭주 셀 입자 F 이력 vs 정상·희소 정상·정답.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader
df -h /home | tail -1
O=$W/lbours/fem_fh_lego; mkdir -p $O
sync_r2() { rclone copy $O r2:storage/result/anchorflow/lbours/fem_fh_lego -q; }
export EX_ROOT=fem_fh_lego EX_NOWAIT=1
R="bash exe/fem_ex_run.sh"
LQ="--mode lag --opt --mass tet --no_dump --lagdiag --lagdump 3 --fhist --floor vel --N 192"
# 연기 시험 (3 프레임) -> 통계 스크립트까지 통과해야 본 실행
$R fh_smoke 4 $LQ --frames 3 < /dev/null > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=4 python -u exe/fem_fh_stats.py $O/fh_smoke --W 2 < /dev/null > $O/fh_smoke_stats.log 2>&1 || { tail -30 $O/fh_smoke_stats.log; touch $O/FH_SMOKE_FAIL; sync_r2; exit 1; }
tail -12 $O/fh_smoke_stats.log
echo "[연기 시험 통과] $(date +%T)"
for g in 0 1 2 3; do
  $R fh_tet_n192_r$g $g $LQ --frames 80 < /dev/null > /dev/null 2>&1 &
done
wait
for g in 0 1 2 3; do
  d=$O/fh_tet_n192_r$g
  ( echo "== $(basename $d)"; cat $d/err_fem.txt
    CUDA_VISIBLE_DEVICES=$g python -u exe/fem_fh_stats.py $d < /dev/null > $d/fh_stats.log 2>&1; echo "rc $?"; tail -40 $d/fh_stats.log ) > $d/fh_stats_sum.txt &
done
wait
cat $O/fh_tet_n192_r?/fh_stats_sum.txt
du -sh $O
sync_r2
touch $O/FH_DONE
echo FH_DONE $(date +%T)
