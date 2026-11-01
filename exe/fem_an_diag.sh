#!/bin/bash
# 절 AN 뉴턴 수렴 원인 가르기 (비측정): lego L000 앞 4 프레임, K 1, 판 하나씩 (가장 한가한 카드 둘에 나눠 순서대로).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
R=/tmp/dkta/lbours/fem_an/${AN_DIAG:-diag1}; M=/tmp/dkta/lbours/fem_an/marks; mkdir -p $R $M
GS=($(for i in $(seq 10); do nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits; sleep 1; done | awk -F', ' '$1>0{s[$1]+=$2} END{for (k in s) print s[k], k}' | sort -n | head -2 | awk '{print $2}'))
B="--data /home/dkta/work/one/L000/L000.pt --tag L000 --frames 4 --K 1 --integ mid --gfin --diag --nt_tol 1e-12"
declare -A VS=( [A_full]="--pc mg --newton 10 --cg 400 --cg_tol 1e-10" [B_noproj]="--pc mg --newton 10 --cg 400 --cg_tol 1e-10 --no_proj"
  [C_nofloor]="--pc mg --newton 10 --cg 400 --cg_tol 1e-10 --floor none" [D_be]="--pc mg --newton 10 --cg 400 --cg_tol 1e-10 --integ be"
  [E_ref]="--pc mg --newton 10 --cg 100 --cg_tol 1e-4" [F_jacfull]="--pc jacobi --newton 10 --cg 1500 --cg_tol 1e-10" )
run() { G=$1; shift; for v in "$@"; do O=$R/$v; mkdir -p $O; echo "카드 $G" > $O/card.txt
  CUDA_VISIBLE_DEVICES=$G python -u exe/fem_an_sim.py --out $O $B ${VS[$v]} < /dev/null > $O/log.txt 2>&1
  python exe/fem_an_diag_table.py $O > $O/diag.md 2>&1; rclone copy $O r2:storage/result/anchorflow/lbours/fem_an/${AN_DIAG:-diag1}/$v -q; done; }
run ${GS[0]} ${AN_V1:-A_full C_nofloor D_be} > /dev/null 2>&1 &
run ${GS[1]:-${GS[0]}} ${AN_V2:-E_ref B_noproj F_jacfull} > /dev/null 2>&1 &
wait
for v in A_full B_noproj C_nofloor D_be E_ref F_jacfull; do echo "## $v"; cat $R/$v/diag.md; done > $M/${AN_DIAG:-diag1}.md
rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
