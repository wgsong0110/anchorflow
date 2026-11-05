#!/bin/bash
# 절 CM8 씬 파일 (사용자 승인 11-05 정의 변경 둘 중 하나를 목록 인자 --v=a|b 로 고름; 나머지는 절 CM3·CM4 채택 공통 설정 exe/fem_cm3_scn.sh 그대로):
#   --v=a : 모든 바닥 씬 (lego 21·낙하 drop1·wolf) 바닥 = 장벽 대신 가우스 점 고착 (재질 판 cm6 stk 와 같은 정의; lego 도 같은 rt_floor_q·rt_floor_n 경로) -> FB = --floor vel
#   --v=b : 장벽 바닥 유지 + 시작점 섞기 β = 장벽 에너지가 걸음 시작 위치 (U) 의 장벽 에너지보다 크지 않은 가장 큰 2^-k (--y0eb)
# ficus·bread 는 바닥 없음 (--floor none) 이라 두 변형 모두 cm4 와 같은 명령.
source exe/fem_cm3_scn.sh
FB0=$FB
eval "$(declare -f cm_scene | sed '1s/cm_scene/cm_scene_0/')"
cm_scene() { local t=$1; shift; local a=() x v=
  for x in "$@"; do case $x in --v=*) v=${x#--v=};; *) a+=("$x");; esac; done
  case $v in a) FB="--floor vel";; b) FB="$FB0 --y0eb";; *) FB=$FB0;; esac
  cm_scene_0 $t "${a[@]}"; FB=$FB0; true; }
