#!/bin/bash
# 사용자 10-29 "발산한 노드가 질량이 작아서인가, 주변 셀이 찌그러져 힘이 폭발해서인가 -- 발산 셀과 정상 셀의 질량·힘 비교":
#   (1) 절 Q.5/R 진단 설정 그대로 (재평균 없는 lag 판, --mass tet = 셀 집중 m_T/4, 속도 바닥, 이동 제한 끔, N192) 4 회 (원자 합 순서로 발산 프레임이 갈린다),
#   (2) 절 T 판 그대로 (--p2g bs --bs_mesh occ, 제한 끔, N192, G2P bary / bs) 각 1 회 -- 진단 루프 (--lagdiag) 로 같은 _step_lag 커널.
#   모두 --lagdump 3 (10-30 확장: 서브스텝별 내부력 f·F̄_T·∇λ·m_T 추가, 계산 불변) -> exe/fem_dv_stats.py (GPU) 로 폭주 vs 정상 통계. 비측정, 카드마다 하나.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader
O=$W/lbours/fem_dv_lego; mkdir -p $O
export EX_ROOT=fem_dv_lego EX_NOWAIT=1
R="bash exe/fem_ex_run.sh"
LQ="--mode lag --opt --mass tet --no_dump --lagdiag --lagdump 3 --floor vel --N 192"
LT="--mode lag --opt --p2g bs --bs_mesh occ --no_dump --floor vel --rstat --bsdiag -1 --lagdiag --lagdump 3 --N 192"
# 연기 시험 (3 프레임) -> 통계 스크립트까지 통과해야 본 실행
$R dv_smoke_tet 6 $LQ --frames 3 < /dev/null > /dev/null 2>&1
CUDA_VISIBLE_DEVICES=6 python -u exe/fem_dv_stats.py $O/dv_smoke_tet < /dev/null > $O/dv_smoke_stats.log 2>&1 || { tail -20 $O/dv_smoke_stats.log; touch $O/DV_SMOKE_FAIL; rclone copy $O r2:storage/result/anchorflow/lbours/fem_dv_lego -q; exit 1; }
tail -15 $O/dv_smoke_stats.log
echo "[연기 시험 통과] $(date +%T)"
for g in 0 1 2 3; do
  $R dv_tet_n192_r$g $g $LQ --frames 80 < /dev/null > /dev/null 2>&1 &
done
$R dv_bsT_gb_n192 4 $LT --g2p bary < /dev/null > /dev/null 2>&1 &
$R dv_bsT_gs_n192 5 $LT --g2p bs < /dev/null > /dev/null 2>&1 &
wait
for d in $O/dv_tet_n192_r0 $O/dv_tet_n192_r1 $O/dv_tet_n192_r2 $O/dv_tet_n192_r3 $O/dv_bsT_gb_n192 $O/dv_bsT_gs_n192; do
  echo "== $(basename $d)"; cat $d/err_fem.txt
  CUDA_VISIBLE_DEVICES=0 python -u exe/fem_dv_stats.py $d < /dev/null > $d/dv_stats.log 2>&1; echo "rc $?"; tail -14 $d/dv_stats.log
done
rclone copy $O r2:storage/result/anchorflow/lbours/fem_dv_lego -q
touch $O/DV_DONE
echo DV_DONE $(date +%T)
