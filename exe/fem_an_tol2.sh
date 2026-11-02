#!/bin/bash
# 절 AN.14 허용치 훑기 2 차 (규칙 고침: 기준 척도 F = |T f(y0)| 걸음 시작점 탄성·접촉 힘, 뉴턴 |g_k| <= εN F, PCG |r| <= εP min(|g_k|, F)) -- oct1 뒤.
# 결정적, 상한 6 x 30 (기준 판은 tol1 의 beref·hhref = 8 x 40, 허용치에 안 닿아 상한까지 -- 규칙과 무관), 그리고 실제 시간 상한 판 (lego 1 x 5, ficus 2 x 10).
cd "$(dirname "$0")/.." || exit 1
export PATH=$HOME/bin:$PATH
M=/dev/shm/dkta/lbours/fem_an/marks; mkdir -p $M /dev/shm/dkta/an_lists
while [ ! -f $M/oct1.END ]; do sleep 60; done
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mg_lreuse --det"
L=/dev/shm/dkta/an_lists/tol2.txt; : > $L
for t in ficus 0001 0016 L000; do for ig in be hh; do [ $ig = be ] && I="--integ be" || I="--ga hht --rho 0.7"
  for np in 1e-1:1e-1 1e-2:1e-1 1e-3:1e-1 1e-2:1e-2; do n=${np%:*}; p=${np#*:}; echo "${ig}n${n#1e-}p${p#1e-} $t $B $I --newton 6 --cg 30 --tolN $n --tolP $p" >> $L; done
  if [ $t = ficus ]; then echo "${ig}cap $t $B $I --newton 2 --cg 10 --tolN 1e-2 --tolP 1e-1" >> $L; else echo "${ig}cap $t $B $I --newton 1 --cg 5 --tolN 1e-2 --tolP 1e-1" >> $L; fi
done; done
wc -l $L
AN_CARDS="1 2 4 6" bash exe/fem_an_pool.sh tol2 $L > $M/tol2.log 2>&1
/home/dkta/.conda/envs/af/bin/python exe/fem_an_tol_table.py /dev/shm/dkta/lbours/fem_an/tol2 > $M/tol2_table.md 2>&1
echo "[tol2 끝] $(date '+%F %T')" > $M/tol2.END; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
