#!/bin/bash
# 절 AN.err 작은 Δt 진단 2 (비측정): 낙하 블록 앞 8 프레임 -- 이탈 해제 끔 / 후퇴 오일러로 큰 K, 관성 질량 하한
cd "$(dirname "$0")/.." || exit 1
L=/tmp/dkta/anerr_diag2.txt
B="--frames 8 --ls_diff --en"
cat > $L <<EOT
y_mK8mfl drop1 --K 8 --integ mid --pc mg --newton 6 --cg 30 --mfl 1 $B
y_mK32nvo drop1 --K 32 --integ mid --pc mg --newton 4 --cg 20 --no_vfl_out $B
y_bK32 drop1 --K 32 --integ be --pc mg --newton 4 --cg 20 $B
y_bK267 drop1 --K 267 --integ be --pc jacobi --newton 2 --cg 10 $B
y_mK267nvo drop1 --K 267 --integ mid --pc jacobi --newton 2 --cg 10 --no_vfl_out $B
y_mK267mfl drop1 --K 267 --integ mid --pc jacobi --newton 2 --cg 10 --mfl 1 $B
EOT
ERR_PAR=6 bash exe/fem_anerr_pool.sh diag2 $L
