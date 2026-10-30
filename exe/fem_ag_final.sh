#!/bin/bash
# 절 AG 최종 (사용자 10-30 공통 설정: 확정 플래그 + 소성·점성·파괴 + h_s 6.25e-5 고정, N = frame_dt/h_s): 단독 측정 3 회 (GPU0 + gpu0.lock, 다른 카드 빈 상태) ->
# 갈래 (셀 return mapping 없음 -- 프레임 끝 입자만) 단독 측정 -> 렌더 (카드 1~6) -> 수박 파괴 진단 (고정 격자가 갈라진 양쪽을 잇는 비율).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
OS=/tmp/dkta/lbours/fem_ag_sum; mkdir -p $OS
MAIN="hotdog:267 bread:160 wolf:640 wm:480"
AG_TIMED="$MAIN" bash exe/fem_ag_timed.sh
AG_TIMED="hotdog:267 bread:160" AG_SFX=_nocell AG_X="--pl_cell none" bash exe/fem_ag_timed.sh
AG_TIMED="$MAIN" AG_CARDS="1 2 3 4" bash exe/fem_ag_render.sh &
AG_TIMED="hotdog:267 bread:160" AG_SFX=_nocell AG_CARDS="5 6" bash exe/fem_ag_render.sh &
CUDA_VISIBLE_DEVICES=0 python -u exe/fem_ag_crack.py --gt /tmp/dkta/ag_data/wm.pt --node_h 0.0094281 --out $OS/crack_wm.json > $OS/crack_wm.log 2>&1
wait
rclone copy $OS r2:storage/result/anchorflow/lbours/fem_ag_sum -q
echo "AG_FINAL_DONE $(date +%T)"
