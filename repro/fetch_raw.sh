#!/bin/bash
# 논문 표·그림 집계에 필요한 R2 원자료 중 작은 파일 (json·log·cmd.txt·commit.txt·지표) 만 받는다. 큰 덤프 (npz·pt·h5)·png·mp4 는 받지 않음.
#   bash repro/fetch_raw.sh            -> ${RAW:-~/.cache/fusedgs_raw}/<r2:storage/result/anchorflow 아래 같은 경로>
#   FETCH_N1=1   : 안정성 우리 판 N=1 세 번 (a·b·c) 의 res.npz 도 받음 (약 250 MB; stab_final_1115.tex 의 "N=1 세 번" 표만 씀, 논문 숫자와 무관)
#   FETCH_PNG=1  : 빵 찢김 그림 (bread_tear.png) 렌더 PNG 도 받음 (results_all/tools/bread_tear.py 입력, 약 수백 MB)
# 받은 뒤: bash repro/make_tables.sh
set -euo pipefail
RAW=${RAW:-$HOME/.cache/fusedgs_raw}; R=${R2ROOT:-r2:storage/result/anchorflow}
mkdir -p "$RAW"
RC="rclone copy -q --transfers 16 --checkers 32"
cp_() {   # cp_ <R2 상대 폴더> <include 패턴...>
  local src=$1; shift; local inc=(); for p in "$@"; do inc+=(--include "$p"); done
  $RC "$R/$src" "$RAW/$src" "${inc[@]}"; echo "  $src"
}
echo "[1/5] 안정성 (stab_avg)"
for s in lego pillow2sofa ship_plastic_lego mic_metal_lego; do
  cp_ lbours/stabfinal/res/$s "/an.json" "/extra.json" "/plan_*.json" "/config.json" "/vis/vis.json" "/speed/*/log.txt" "/hex/*/log.txt"
done
$RC "$R/lbours/stabhex" "$RAW/lbours/stabhex" --include "/stabhex_an.json"; echo "  lbours/stabhex/stabhex_an.json"
$RC "$R" "$RAW" --include "/stab_scan_table.json" --include "/stab_metrics_20261014.json"; echo "  stab_scan_table.json (레고 PG·i-PG 11-12 값 대조용)"
if [ "${FETCH_N1:-0}" = 1 ]; then
  for st in lego:_f5 pillow2sofa:_f5 ship_plastic_lego:_f7 mic_metal_lego:_f7; do s=${st%%:*}; t=${st#*:}
    for r in a b c; do cp_ lbours/stabfinal/res/$s/hex/${s}_N1_$r$t "/res.npz" "/log.txt" "/FIN"; done
  done
fi
echo "[2/5] 파레토·ablation·강인성 (pareto_mean, abl_design, abl_robust, robust_bar)"
cp_ lbours/paper_remeasure2/final2 "/all.json"
cp_ lbours/final_pa/bl "/pareto_bl.json"
cp_ lbours/final_pa/blx "/pareto_blx.json" "/*/g/*_dump/phys.json" "/*/g/*_dump/vis_ffx.json" "/*/g/*_dump/vis.json"
RUNF=("/*/metrics.json" "/*/cmd.txt" "/*/commit.txt" "/*/eig.json")
cp_ lbours/final_fix "/*.json" "/metrics_def.md" "/abl_eig/*/eig.json" \
    "/phys/*/phys.json" "/phys/*/vis.json" "/phys/*/vis_ffx.json" \
    "/bl_mpm/*.json" "/bl_mpm/*/phys.json" "/bl_mpm/*/vis.json" "/bl_mpm/*/chosen.txt" "/bl_mpm/*/*/err.json" "/bl_mpm/*/*/phys.json" "/bl_mpm/*/*/vis.json"
for d in runs runs_nd runs_wolfN; do cp_ lbours/final_fix/$d "${RUNF[@]}"; done
echo "[3/5] ablation 1^3 시간 (gq_justify)"
cp_ lbours/gq_justify/runs "/*_base_time_r*/gq.json" "/*_base_time_r*/cmd.txt" "/*_base_time_r*/commit.txt"
echo "[4/5] 표현력 (rep_l2)"
cp_ lbours/rep_final1115 "/*/*/*.log" "/*/*/cmd.txt"
cp_ lbours/repflow_hex/main "/wolf_hex.log" "/wolf_hex.cmd.txt" "/cmd.txt"
cp_ archive_20261009/r10 "/wolf_phystwin_k*.txt"
cp_ lbours/repflow_vrgs/main "/wolf_vrgsx.log" "/cmd.txt"
cp_ lbours/repflow1115/sg/main/wolf "/sg_wolf.log" "/cmd.txt"
if [ "${FETCH_PNG:-0}" = 1 ]; then
  echo "[5/5] 빵 찢김 렌더 PNG"
  B=$RAW/bread_tear_png
  for x in final_fix/phys/mpm_bread:phys_mpm_bread final_fix/phys/ours_bread_def_bs2:phys_ours_bread_def_bs2 \
           final_pa/blx/A/g/phystwin_bread_n200_dump:blx_phystwin_bread_n200_dump final_pa/blx/A/g/vrgs_bread_n100_dump:blx_vrgs_bread_n100_dump \
           final_pa/blx/SGB/g/springgaus_bread_n200_dump:blx_springgaus_bread_n200_dump; do
    $RC "$R/lbours/${x%%:*}/png" "$B/${x#*:}"; echo "  ${x%%:*}/png"
  done
else
  echo "[5/5] 빵 찢김 PNG 건너뜀 (FETCH_PNG=1 이면 받음)"
fi
du -sh "$RAW"
