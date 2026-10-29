#!/bin/bash
# 절 T 4 차 (사용자 10-29: 추가 스윕 없이 렌더만): lego 이동 제한 판 N192 (G2P B-스플라인, box 0.3 -- 스윕 5/5 완주) F 덤프 + 3DGS 렌더, GPU0 단독
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
LB="--mode lag --opt --p2g bs --bs_mesh occ --floor vel --rstat --bsdiag -1 --g2p bs --lim 0.3 --lim_mode box"
for a in 1 2 3; do
  SHAPE=lego EX_ROOT=fem_t_lego bash exe/fem_ex_final.sh f_l_gsL_n192_F$a 0 $LB --dump_f32 --N 192
  [ -f $W/lbours/fem_t_lego/f_l_gsL_n192_F$a/vj/RENDER_DONE ] && { echo "[최종] 시도 $a 에서 완주·렌더"; break; }
  echo "[최종] 시도 $a 발산/실패"
done
rclone copy $W/lbours/fem_t_lego r2:storage/result/anchorflow/lbours/fem_t_lego -q
echo T4_DONE $(date +%T)
