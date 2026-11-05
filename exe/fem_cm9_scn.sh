#!/bin/bash
# 절 CM9 씬 파일 (사용자 승인 11-05 정의 변경 --gq_mass 는 목록 인자로 켜고 끔): 절 CM8 씬 (exe/fem_cm8_scn.sh, --v=b 가 채택 판 (B) 장벽 + --y0eb) 에 수박 wm 을 더함.
#   wm : 지금 수박 판 = 절 AS 전체 판 (절 AT at2 와 같은 명령에서 진단 --nanchk·--at_diag 만 뺌) -- 옛 재질 실행기 exe/fem_ao_rt_sim.py (lib/anchorflow/fem_ao_rt.py),
#        GF 바닥 평면 z 0.02 미끄럼, 셀 1.5, HHT ρ 0.7 K 3 뉴턴 2 x PCG 5, 허용치, 시작점 관성 예측, 거의 빈 노드 문턱 1e-3 x 중앙값, --ls_inv --y0u --y0l --fq_rl --ls_end.
#        (새 실행기 exe/fem_an_rt_sim.py 에는 바닥 평면·미끄럼 바닥·AS 필터가 없다 -- 그래서 수박은 지금 판 그대로 옛 실행기에서 켜고 끔)
source exe/fem_cm8_scn.sh
WMD=/tmp/dkta/ag_data; mkdir -p $WMD 2>/dev/null
WMA="--cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --K 3 --newton 2 --cg 5 --ls_inv --y0u --y0l --fq_rl --ls_end"
eval "$(declare -f cm_scene | sed '1s/cm_scene/cm_scene_8/')"
eval "$(declare -f cm_data | sed '1s/cm_data/cm_data_8/')"
cm_data() { case $1 in
    wm) [ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; };;
    *) cm_data_8 "$@";;
  esac; }
cm_scene() { local t=$1
  if [ $t = wm ]; then shift; cm_data wm
    SIM=exe/fem_ao_rt_sim.py; DA="--data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --floor_slip $WMA"; XA="$*"
  else cm_scene_8 "$@"; fi; }
