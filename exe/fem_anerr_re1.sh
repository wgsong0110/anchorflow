#!/bin/bash
# 절 AN.err 다시 띄움 1 (노드 /tmp 가득 참으로 실패한 판; 출력 뿌리 /dev/shm): 진단 2 남은 판 + 원본 f64 경로 K 8, K 1 요인 묶음 lego 남은 판. 두 풀 동시에 (카드는 빈 카드 판정·잡음 표식을 함께 씀)
cd "$(dirname "$0")/.." || exit 1
B="--frames 8 --ls_diff --en"
cat > /tmp/dkta/anerr_re1a.txt <<EOT
y_bK32 drop1 --K 32 --integ be --pc mg --newton 4 --cg 20 $B
y_bK267 drop1 --K 267 --integ be --pc jacobi --newton 2 --cg 10 $B
y_mK267nvo drop1 --K 267 --integ mid --pc jacobi --newton 2 --cg 10 --no_vfl_out $B
y_mK267mfl drop1 --K 267 --integ mid --pc jacobi --newton 2 --cg 10 --mfl 1 $B
EOT
echo "x_mK8slow drop1 --K 8 --integ mid --pc mg --newton 10 --cg 100 --cg_tol 1e-8 --nt_tol 1e-8 --frames 8 --en --slow" > /tmp/dkta/anerr_re1b.txt
L=/tmp/dkta/anerr_re1c.txt; : > $L
K1="--K 1 --pc mg --newton 6 --cg 30 --ls_diff --en"
for t in 0001 0003 0016 0017 0018 0005 0013 L000; do
  for v in "g_mid:--integ mid" "g_be:--integ be" "g_bdf2:--integ bdf2" "p_mid:--integ mid --quad part" "g_flit:--integ mid --fl_iter" "g_flx:--integ mid --fl_src part --fl_npd 0.005" "g_nvo:--integ mid --no_vfl_out"; do
    [ "${v%%:*}_$t" = g_be_0001 ] && continue
    echo "${v%%:*}_K1 $t ${v#*:} $K1" >> $L; done; done
ERR_PAR=4 bash exe/fem_anerr_pool.sh diag2 /tmp/dkta/anerr_re1a.txt &
sleep 20; ERR_PAR=1 bash exe/fem_anerr_pool.sh diag /tmp/dkta/anerr_re1b.txt &
sleep 20; ERR_PAR=6 bash exe/fem_anerr_pool.sh k1fac $L &
wait
