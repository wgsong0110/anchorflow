# 절 CM5 진단용 씬 파일: 절 CM3 공통 (exe/fem_cm3_scn.sh) 그대로 + 목록 인자 --U0 이면 시작점 관성 예측 (--ga_y0ut) 만 뺌 (진단 전용, 시작점 U)
source exe/fem_cm3_scn.sh
eval "$(declare -f cm_scene | sed '1s/cm_scene/cm_scene_0/')"
cm_scene() { local t=$1; shift; local u=0 a=() x; for x in "$@"; do if [ "$x" = --U0 ]; then u=1; else a+=("$x"); fi; done
  cm_scene_0 $t "${a[@]}"; [ $u = 1 ] && DA=${DA// --ga_y0ut/}; true; }
