#!/bin/bash
# R2 원자료 사본 (repro/fetch_raw.sh) -> 논문 repo 의 원본 표 tex -> results_all/tools/rebuild.sh -> paper_tables.py -> robust_bar.py 를 한 번에.
#   bash repro/fetch_raw.sh && bash repro/make_tables.sh
#   RAW   = 원자료 사본 (기본 ~/.cache/fusedgs_raw)
#   PAPER = 논문 repo 뿌리 (기본 /home/wgsong/workspace/FusedGS/paper; 사본에서 시험하려면 rsync -a --exclude .git 한 폴더)
#   T     = 중간 폴더 (기본 $PAPER/results_all/build/repro, git 제외)
# 쓰는 원본 표 (논문 repo 뿌리): stab_final_1115.tex (+ _rank.tex), pareto_final_1115_{table,phys,vis}.tex, pareto_mpmbl_cells.json,
#   ablation_final_1115_{tables,metrics}.tex, rep_final_1115.tex, results_all/rep_table.tex
# 그 뒤 논문 표·그림: cvpr2027/tab/*.tex, cvpr2027/fig/{pareto_mean,robust_bar}.pdf (bread_tear.png 는 $RAW/bread_tear_png 가 있을 때만, FETCH_PNG=1)
# 손 값·손 편집이 있던 자리 (README 4 절 '주의') 는 모두 생성기 안 코드 또는 repro/data/*.json (출처 적힘) 으로 옮겼다.
set -euo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RAW=${RAW:-$HOME/.cache/fusedgs_raw}; PAPER=$(cd "${PAPER:-/home/wgsong/workspace/FusedGS/paper}" && pwd); T=${T:-$PAPER/results_all/build/repro}
export LB=$RAW/lbours
[ -f "$LB/stabfinal/res/lego/an.json" ] || { echo "원자료 없음: $RAW (먼저 bash repro/fetch_raw.sh)"; exit 1; }
mkdir -p "$T"; cd "$W"
echo "[1/4] 안정성 -> stab_final_1115.tex"
C=$T/stab_col; rm -rf "$C"; mkdir -p "$C"
for s in lego pillow2sofa ship_plastic_lego mic_metal_lego; do ln -s "$LB/stabfinal/res/$s" "$C/$s"; done
cp exe/stabfinal_notes.tex "$C/notes.tex"; cp exe/stabfinal_settings.tex "$C/settings.tex"
if [ -f "$LB/stabfinal/res/lego/hex/lego_N1_a_f5/res.npz" ]; then python3 exe/stabfinal_n1.py "$LB/stabfinal/res" "$C/n1.json" > "$T/stab_n1.log"
else echo "  (N=1 세 번 res.npz 없음 -> 그 표 빠짐; FETCH_N1=1 로 받으면 들어감)"; fi
python3 - "$RAW" <<'P'      # 레고 PG·i-PG 11-12 값 (원자료 없음) 을 남아 있는 정수 스캔 1 회 기록과 대조
import json, sys, os
L = json.load(open("repro/data/stab_lego_pg_ipg_1112.json")); f = f"{sys.argv[1]}/stab_scan_table.json"
if os.path.exists(f):
    sc = json.load(open(f))["lego"]
    for m in ("pg", "ipg"):
        assert sc[L[m]["scan_key"]]["nmin"] == L[m]["N_min"], (m, sc[m]["nmin"], L[m]["N_min"])
    print("  레고 PG·i-PG N_min (11-12, 원자료 없음) = 정수 스캔 기록과 같음:", {m: L[m]["N_min"] for m in ("pg", "ipg")})
P
python3 exe/stabfinal_rank.py "$C" "$PAPER" > "$T/stab_rank.log"
SF_OVERWRITE=1 python3 exe/stabfinal_tab.py "$C" "$LB/stabhex/stabhex_an.json" "$PAPER" > "$T/stab_tab.log"
echo "[2/4] 파레토·ablation -> pareto_final_1115_{table,phys,vis}.tex, pareto_mpmbl_cells.json, ablation_final_1115_{tables,metrics}.tex"
PAPER=$PAPER T=$T/cm192 FFX_NOTEX=1 bash exe/ffx_cm192_refresh.sh
echo "[3/4] 표현력 -> rep_final_1115.tex, results_all/rep_table.tex"
OLD=(); [ -f "$PAPER/repflow_results.tex" ] && OLD=(--old "$PAPER/repflow_results.tex")   # wolf 열을 옛 표와 대조 (있을 때만)
python3 exe/make_rep_final_tex.py --raw "$RAW" "${OLD[@]}" --pt best --out "$PAPER/rep_final_1115.tex" --table_out "$PAPER/results_all/rep_table.tex" > "$T/rep.log"
echo "[4/4] 논문 표·그림 (results_all/tools)"
PAPER=$PAPER/ GQJ_RUNS=$LB/gq_justify/runs bash "$PAPER/results_all/tools/rebuild.sh" | tee "$T/rebuild.log" | tail -3
PAPER=$PAPER/ python3 "$PAPER/results_all/tools/paper_tables.py"
PAPER=$PAPER/ python3 "$PAPER/results_all/tools/robust_bar.py" | tee "$T/robust_bar.log"
if [ -d "$RAW/bread_tear_png" ]; then
  PAPER=$PAPER/ python3 "$PAPER/results_all/tools/bread_tear.py" "$RAW/bread_tear_png" && cp "$PAPER/fig/bread_tear_strip_1116.png" "$PAPER/cvpr2027/fig/bread_tear.png"
fi
echo "끝: $PAPER/cvpr2027/tab/*.tex, cvpr2027/fig/{pareto_mean,robust_bar}.pdf (중간 $T)"
