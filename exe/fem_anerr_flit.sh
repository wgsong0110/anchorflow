#!/bin/bash
# 절 AN.err 고착 상태를 뉴턴마다 다시 재는 판 (탄성 반응을 본 위치로 이탈·고착을 정함) -- 후퇴 오일러 K 1·267 낙하, lego 0001 K 1, 중점 + 질량 하한 K 267 낙하 (비측정 80 프레임)
cd "$(dirname "$0")/.." || exit 1
mkdir -p /dev/shm/dkta; L=/dev/shm/dkta/anerr_flit.txt
S="--pc mg --newton 6 --cg 30 --ls_diff --en"; J="--pc jacobi --newton 2 --cg 10 --ls_diff --en"
cat > $L <<EOT
bi_be_K1 drop1 --K 1 --integ be --fl_iter $S
bi_be_K267 drop1 --K 267 --integ be --fl_iter $J
bi_be_K1 0001 --K 1 --integ be --fl_iter $S
bim_mid_K267 drop1 --K 267 --integ mid --mfl 1 --fl_iter $J
EOT
ERR_PAR=4 bash exe/fem_anerr_pool.sh flit $L
