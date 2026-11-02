#!/bin/bash
# 절 AN.err 후퇴 오일러 K 267 lego 0005 (만족 대조의 시간 몫, 비측정 80 프레임)
cd "$(dirname "$0")/.." || exit 1
mkdir -p /dev/shm/dkta; L=/dev/shm/dkta/anerr_k267b.txt
echo "g_be_K267 0005 --K 267 --integ be --pc jacobi --newton 2 --cg 10 --ls_diff --en" > $L
ERR_PAR=1 bash exe/fem_anerr_pool.sh kbe $L
