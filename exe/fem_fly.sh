#!/bin/bash
# 사용자 10-29 "발산 프레임에서 노드가 확 날아가는 것을 전후 이미지로": 절 Q.5 진단 설정 그대로 (재평균 없는 lag 판, --mass tet, 속도 바닥) 를
#   --lagdump 3 (마지막 3 프레임의 서브스텝별 노드 U·v·det A_T·접촉 α 덤프, 계산 불변) 으로 다시 돈다. 비측정, 카드마다 하나.
#   같은 설정도 원자 합 순서로 발산 여부가 갈리므로 (절 Q.3·Q.5) N192 frac 을 카드 4 장에, Q.5 의 나머지 바닥 판 (N96·N384·full N192) 을 1 장씩.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader
LB="--mode lag --opt --mass tet --no_dump --lagdiag --lagdump 3 --floor vel"
export EX_ROOT=fem_fly_lego EX_NOWAIT=1
R="bash exe/fem_ex_run.sh"
for g in 0 1 2 3; do
  $R fly_vel_n192_r$g $g $LB --N 192 --frames 80 < /dev/null > /dev/null 2>&1 &
done
$R fly_vel_n96 4 $LB --N 96 --frames 30 < /dev/null > /dev/null 2>&1 &
$R fly_vel_n384 5 $LB --N 384 --frames 45 < /dev/null > /dev/null 2>&1 &
$R fly_velfull_n192 6 $LB --vfl_mode full --N 192 --frames 40 < /dev/null > /dev/null 2>&1 &
wait
for d in $W/lbours/fem_fly_lego/fly_*; do echo "== $(basename $d)"; grep -a "롤아웃\]\|발산\|lag 덤프\]" $d/v_fem.log | tail -3; done
rclone copy $W/lbours/fem_fly_lego r2:storage/result/anchorflow/lbours/fem_fly_lego -q
touch $W/lbours/fem_fly_lego/FLY_DONE
echo FLY_DONE $(date +%T)
