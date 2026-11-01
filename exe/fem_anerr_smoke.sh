#!/bin/bash
# 절 AN.err 첫 시험: 한 카드에서 차례로 (각 4 프레임) -- 가우스 MG K1, 입자 적분점, BDF2, 입자 바닥+비관통, 야코비 K 267, 낙하 블록 --en
cd "$(dirname "$0")/.." || exit 1
cat > /tmp/dkta/anerr_smoke.txt <<'EOT'
EOT
for l in "g_m1 0001 --pc mg --K 1 --integ mid --newton 3 --cg 20 --en" "p_m1 0001 --pc mg --K 1 --integ mid --newton 3 --cg 20 --quad part --en" \
         "g_b2 0001 --pc mg --K 2 --integ bdf2 --newton 3 --cg 20" "g_fl 0001 --pc mg --K 1 --integ be --newton 3 --cg 20 --fl_src part --fl_npd 0.005" \
         "g_j267 0001 --pc jacobi --K 267 --integ mid --newton 2 --cg 10" "p_m2 drop1 --pc mg --K 2 --integ mid --newton 3 --cg 20 --quad part --en --fl_npd 0.005"; do echo "$l" >> /tmp/dkta/anerr_smoke.txt; done
ERR_FR=4 ERR_PAR=1 bash exe/fem_anerr_pool.sh smoke /tmp/dkta/anerr_smoke.txt
