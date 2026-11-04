#!/bin/bash
# 절 CM6 진단용 씬 파일: 절 CM3 공통 (exe/fem_cm3_scn.sh) 그대로 + 목록 인자
#   --no=<플래그>  : DA 에서 그 플래그 (뒤에 수 값이 있으면 값까지) 를 뺌 (인자 하나씩 바꾸기)
#   --noba         : 공통 묶음 BA·장벽 바닥을 뺌 (예전 재질 판 비교; CM_NOBA 와 같음)
source exe/fem_cm3_scn.sh
eval "$(declare -f cm_scene | sed '1s/cm_scene/cm_scene_0/')"
cm_scene() { local t=$1; shift; local rm=() nb=0 a=() x
  for x in "$@"; do case $x in --no=*) rm+=("${x#--no=}");; --noba) nb=1;; *) a+=("$x");; esac; done
  cm_scene_0 $t "${a[@]}"
  [ $nb = 1 ] && { DA=${DA//$BA/}; DA=${DA// --fl_bar --fl_bst 1/}; }
  for x in "${rm[@]}"; do DA=$(echo " $DA " | sed -E "s/ $x( [0-9][0-9.e+-]*)? / /g; s/^ +//; s/ +$//"); done
  true; }
