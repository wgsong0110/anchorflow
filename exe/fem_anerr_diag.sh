#!/bin/bash
# 절 AN.err 작은 Δt 접촉 진단 (비측정): 낙하 블록 앞 8 프레임
cd "$(dirname "$0")/.." || exit 1
L=/tmp/dkta/anerr_diag.txt
B="--frames 8 --ls_diff --en"
cat > $L <<EOT
x_mK2 drop1 --K 2 --integ mid --pc mg --newton 6 --cg 30 $B
x_mK4 drop1 --K 4 --integ mid --pc mg --newton 6 --cg 30 $B
x_bK8 drop1 --K 8 --integ be --pc mg --newton 6 --cg 30 $B
x_mK8nvo drop1 --K 8 --integ mid --pc mg --newton 6 --cg 30 --no_vfl_out $B
x_mK8jac drop1 --K 8 --integ mid --pc jacobi --newton 6 --cg 100 $B
x_mK8fx drop1 --K 8 --integ mid --pc mg --newton 6 --cg 30 --fl_src part --fl_npd 0.005 $B
x_mK8slow drop1 --K 8 --integ mid --pc mg --newton 10 --cg 100 --cg_tol 1e-8 --nt_tol 1e-8 --frames 8 --en --slow
EOT
ERR_PAR=6 bash exe/fem_anerr_pool.sh diag $L
