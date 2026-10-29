#!/bin/bash
# 절 Q.5 (사용자 10-29 "왔다 갔다 여러 번 할 필요 없다"): 재평균 없는 lag 판 (프레임당 P2G 1 회 -> 메시 명시적 서브스텝 N -> G2P 1 회) lego 발산 원인 진단.
#   노드 질량·운동량 = 셀 (m_T, m_T v̄_T) 을 꼭짓점에 ¼ (--mass tet, 사용자 의도 정의). --lagdiag 메시 진단 (질량 분포·점유율·국소 안정 비·접촉). 비측정, 카드마다 하나.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline)"
LB="--mode lag --opt --mass tet --no_dump --lagdiag"
export EX_ROOT=fem_q_lego EX_NOWAIT=1
R="bash exe/fem_ex_run.sh"
$R d_l_vel_n96 0 $LB --floor vel --N 96 --frames 30 < /dev/null > /dev/null 2>&1 &
$R d_l_vel_n192 1 $LB --floor vel --N 192 --frames 40 < /dev/null > /dev/null 2>&1 &
$R d_l_vel_n384 2 $LB --floor vel --N 384 --frames 45 < /dev/null > /dev/null 2>&1 &
$R d_l_none_n192 3 $LB --floor none --N 192 --frames 30 < /dev/null > /dev/null 2>&1 &
$R d_l_velfull_n192 4 $LB --floor vel --vfl_mode full --N 192 --frames 40 < /dev/null > /dev/null 2>&1 &
wait
for d in $W/lbours/fem_q_lego/d_l_*; do echo "== $(basename $d)"; grep -a "롤아웃\]\|발산" $d/v_fem.log | tail -2; done
rclone copy $W/lbours/fem_q_lego r2:storage/result/anchorflow/lbours/fem_q_lego -q --include "d_l_*/**"
echo Q5_DONE
