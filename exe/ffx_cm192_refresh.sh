#!/bin/bash
# 절 CM192: 표·그림 전부 다시 만들고 컴파일 (로컬). 사용: bash exe/ffx_cm192_refresh.sh [SG pillow2sofa 메모]
set -e
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); P=${PAPER:-/home/wgsong/workspace/FusedGS/paper}; T=/tmp/cm192
cd $W
bash exe/ffx_nd2_cache.sh $T/tab > /dev/null
SGNOTE="${1:-}" bash $W/exe/ffx_cm192_gen.sh $P > $T/gen.log 2>&1
python3 exe/ffx_phys_merge.py $T/physm > /dev/null
python3 exe/ffx_phys_tab.py $T/physm $T/pt $P > $T/pt.log 2>&1
python3 exe/ffx_pareto_final.py $T/tab $T/physm $P > $T/rf.log 2>&1
cd $P; for t in pareto_final_1115 ablation_final_1115; do lualatex -interaction=nonstopmode $t.tex > $T/$t.ltx 2>&1 || true; echo "$t $(grep -c '^!' $T/$t.ltx) 오류, $(grep -o 'Output written.*' $T/$t.ltx)"; done
