#!/bin/bash
# 절 AN.14 점유 가중 이식 확인 (--gq_oct 0.1 = 오차 원인 쪽 절 AN.err4 본 묶음 값): 수렴 예산 (8 x 40, 허용치 1e-6 / 1e-3) 결정적 ficus·블록 신축 40 프레임·L000 10 프레임, HHT ρ 0.7 K 2 / 후퇴 K 2.
# 비교: implicit-err oc0 hk10_ficus (HHT 수렴, 주기 0.95 / 진폭 0.85 / 0.818 %), oc1. tol1 묶음이 끝난 뒤 (카드 겹침 방지).
cd "$(dirname "$0")/.." || exit 1
export PATH=$HOME/bin:$PATH
M=/dev/shm/dkta/lbours/fem_an/marks; mkdir -p $M /dev/shm/dkta/an_lists
while [ ! -f $M/tol1.END ]; do sleep 60; done
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mg_lreuse --det --newton 8 --cg 40 --tolN 1e-6 --tolP 1e-3"
L=/dev/shm/dkta/an_lists/oct1.txt
cat > $L <<EOT
hk10 ficus $B --ga hht --rho 0.7 --gq_oct 0.1
bk10 ficus $B --integ be --gq_oct 0.1
hk10 stretch $B --ga hht --rho 0.7 --gq_oct 0.1 --frames 40
hk10s L000 $B --ga hht --rho 0.7 --gq_oct 0.1 --frames 10
hk0s L000 $B --ga hht --rho 0.7 --gq_oct 0 --frames 10
EOT
bash exe/fem_an_pool.sh oct1 $L > $M/oct1.log 2>&1
echo "[oct1 끝] $(date '+%F %T')" > $M/oct1.END; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
