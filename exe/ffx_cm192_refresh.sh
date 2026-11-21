#!/bin/bash
# 절 CM192: 표·그림 전부 다시 만들고 컴파일 (로컬). 사용: bash exe/ffx_cm192_refresh.sh [SG pillow2sofa 메모]
#   PAPER = 논문 폴더 (기본 /home/wgsong/workspace/FusedGS/paper), T = 중간 폴더 (기본 /tmp/cm192),
#   LB = R2 lbours 위치 (기본 FUSE ~/r2/result/anchorflow/lbours; repro/fetch_raw.sh 사본이면 $RAW/lbours), FFX_NOTEX=1 이면 lualatex 건너뜀
set -e
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); P=${PAPER:-/home/wgsong/workspace/FusedGS/paper}; T=${T:-/tmp/cm192}; export T
export LB=${LB:-$HOME/r2/result/anchorflow/lbours}
cd $W; mkdir -p $T/pt
bash exe/ffx_nd2_cache.sh $T/tab > /dev/null
SGNOTE="${1:-}" bash $W/exe/ffx_cm192_gen.sh $P > $T/gen.log 2>&1
python3 exe/ffx_phys_merge.py $T/physm > /dev/null
FPA_SUMMARY=$T/tab/fpa_summary.json python3 exe/ffx_phys_tab.py $T/physm $T/pt $P > $T/pt.log 2>&1
python3 exe/ffx_pareto_final.py $T/tab $T/physm $P > $T/rf.log 2>&1
python3 exe/pareto_mpmbl_cells.py $P > $T/cells.log 2>&1      # [repro] i-PG·Omni 오차·ms·부피 -> $P/pareto_mpmbl_cells.json (논문 repo add_mpmbl.py 입력)
[ "${FFX_NOTEX:-0}" = 1 ] && exit 0
cd $P; for t in pareto_final_1115 ablation_final_1115; do lualatex -interaction=nonstopmode $t.tex > $T/$t.ltx 2>&1 || true; echo "$t $(grep -c '^!' $T/$t.ltx) 오류, $(grep -o 'Output written.*' $T/$t.ltx)"; done
