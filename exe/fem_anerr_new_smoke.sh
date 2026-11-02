#!/bin/bash
# 절 AN.err 2 차 첫 시험 (θ 법·TR-BDF2·질량 하한·고착 평균·적응 걸음; 짧게)
cd "$(dirname "$0")/.." || exit 1
mkdir -p /dev/shm/dkta; L=/dev/shm/dkta/anerr_ns.txt
S="--pc mg --newton 6 --cg 30 --ls_diff --en --frames 4"
cat > $L <<EOT
n_th55 drop1 --K 1 --integ theta --theta 0.55 --mlow 0.1 $S
n_trb drop1 --K 1 --integ trbdf2 --mlow 0.1 --mfl_n 267 $S
n_avg drop1 --K 1 --integ theta --theta 0.6 --fl_avg 8 $S
n_kad drop1 --K 1 --kad 4 --integ theta --theta 0.55 $S
n_trb ficus --K 1 --integ trbdf2 --mlow 0.1 --pc mg --newton 6 --cg 30 --ls_diff --frames 3
EOT
ERR_PAR=3 bash exe/fem_anerr_pool.sh nsmoke $L
