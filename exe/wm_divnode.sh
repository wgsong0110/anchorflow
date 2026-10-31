#!/bin/bash
# 사용자 10-31 "발산한 노드 몇 개의 서브스텝별 속도 그래프 + 평균·분산":
#   수박 = 절 AJ 진단 실행 (exe/fem_dc.sh 의 dc_wm 명령) 그대로 + --divhist 4 (발산 프레임 4 의 서브스텝별 전 노드 |v|·|f|·접촉·질량, 계산 불변).
#   발산 검출로 프레임 4 끝에서 스스로 멈춘다 (절 AJ 와 같음).
#   -> exe/wm_divnode_stats.py (작은 npz) -> R2 lbours/fem_dc/$VT 통째로 (큰 덤프 지우지 않음).
# 비측정, GPU0 은 쓰지 않는다 (gpu0.lock 단독 측정용) -- 카드 1 에서 6 중 빈 것 (사용 < 500 MiB) 이 날 때까지 기다린다.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
source exe/fem_ag_scene.sh
RR=r2:storage/result/anchorflow; VT=${VH_TAG:-dc_wm_vh}; O=/tmp/dkta/lbours/fem_dc/$VT
echo "[커밋] $(git log -1 --oneline) $(date '+%F %T')"
while true; do
  G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1 != 0 && $2 < 500 {print $1}' | head -1)
  [ -n "$G" ] && break
  echo "[대기] 카드 1 에서 6 빈 것 없음 $(date +%T)"; sleep 120
done
echo "[카드] $G $(date +%T)"
ag_scene wm
EX_ROOT=fem_dc bash exe/fem_ab_run.sh ${VH_TAG:-dc_wm_vh} $G $SARGS $REC --lagdiag --divtrack --divhist 4 --dump_f32 --dump_div --N $NC < /dev/null
grep -a "발산 추적\] t 4\|발산 기록\|\[발산\]\|롤아웃\]" $O/v_fem.log | cut -c1-300
python -u exe/wm_divnode_stats.py --dump $O/divhist/f003.pt --out $O/divnode.npz < /dev/null > $O/divnode.log 2>&1; tail -3 $O/divnode.log
rclone copy $O $RR/lbours/fem_dc/$VT -q && rclone check $O $RR/lbours/fem_dc/$VT --one-way --size-only -q && echo "[R2 확인] lbours/fem_dc/$VT"
touch $O/VH_DONE; rclone copy $O $RR/lbours/fem_dc/$VT -q
echo "VH_DONE $(date +%T)"
