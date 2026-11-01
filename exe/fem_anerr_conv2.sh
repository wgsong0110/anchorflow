#!/bin/bash
# 절 AN.err 선탐색 판정 확인 (비측정): 낙하 블록 앞 8 프레임
cd "$(dirname "$0")/.." || exit 1
L=/tmp/dkta/anerr_conv2.txt
cat > $L <<'EOT'
d_j2x10ld_K267 drop1 --frames 8 --K 267 --integ mid --pc jacobi --newton 2 --cg 10 --ls_diff
d_j2x10n_K267 drop1 --frames 8 --K 267 --integ mid --pc jacobi --newton 2 --cg 10 --fast_ls 0
d_m6x30ld_K1 drop1 --frames 8 --K 1 --integ mid --pc mg --newton 6 --cg 30 --ls_diff
d_m6x30ld_K8 drop1 --frames 8 --K 8 --integ mid --pc mg --newton 6 --cg 30 --ls_diff
d_m6x30ld_K32 drop1 --frames 8 --K 32 --integ mid --pc mg --newton 6 --cg 30 --ls_diff
EOT
ERR_PAR=5 bash exe/fem_anerr_pool.sh conv2 $L
