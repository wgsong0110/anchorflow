#!/bin/bash
# 절 AN.14 판정 묶음 j1 (oct1 이 끝난 뒤): 공통 허용치 εN 1e-2 · εP 1e-1 (tol1 근거) + 씬별 상한 (카드 0 단독 16.7 ms 안 최대: lego 1 x 5, ficus 2 x 10; 상한 규칙 = 풀이당 PCG 상한 10 까지 먼저, 그다음 뉴턴 수).
# 판: bej (후퇴 K 2), hhj (HHT ρ 0.7 K 2), bejo·hhjo (+ 점유 가중 κ 0.1, 정의 변경). 궤적: lego L000 + 0000 에서 0019, ficus, 블록 신축·전단 (상한 1x5 와 2x10 둘), 낙하 (1x5·2x10).
# 덧붙임: ficus 상한 모양 1x20·3x6 (같은 시간), 0016 덤프 (원인 진단; 판정 판 + 수렴 기준 판).
# 이어서 단독 시간 tolx2: 낙하·신축 1x5·2x10, ficus 1x20·3x6 (허용치 0 = 최악), ficus 2x10 허용치 판 실제 시간·반복 분포 (후퇴·HHT·점유 가중).
cd "$(dirname "$0")/.." || exit 1
export PATH=$HOME/bin:$PATH
M=/dev/shm/dkta/lbours/fem_an/marks; mkdir -p $M /dev/shm/dkta/an_lists
while [ ! -f $M/oct1.END ]; do sleep 60; done
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mg_lreuse"
T="--tolN 1e-2 --tolP 1e-1"
L=/dev/shm/dkta/an_lists/j1.txt; : > $L
for vv in "bej:--integ be" "hhj:--ga hht --rho 0.7" "bejo:--integ be --gq_oct 0.1" "hhjo:--ga hht --rho 0.7 --gq_oct 0.1"; do v=${vv%%:*}; I=${vv#*:}
  echo "$v ficus $B --det $I $T --newton 2 --cg 10" >> $L
  for k in stretch shear drop1; do echo "$v $k $B --det $I $T --newton 1 --cg 5" >> $L; echo "${v}c2 $k $B --det $I $T --newton 2 --cg 10" >> $L; done
  for t in 0016 L000 $(seq -f %04g 0 15) $(seq -f %04g 17 19); do D=""; [ $t = 0016 ] && D="--dump"; echo "$v $t $B --det $I $T --newton 1 --cg 5 $D" >> $L; done
  echo "${v}s120 ficus $B --det $I $T --newton 1 --cg 20" >> $L
  echo "${v}s306 ficus $B --det $I $T --newton 3 --cg 6" >> $L
done
echo "beref 0016 $B --det --integ be --newton 8 --cg 40 --tolN 1e-6 --tolP 1e-3 --dump" >> $L
echo "hhref 0016 $B --det --ga hht --rho 0.7 --newton 8 --cg 40 --tolN 1e-6 --tolP 1e-3 --dump" >> $L
wc -l $L
AN_CARDS="1 2 4 6" bash exe/fem_an_pool.sh j1 $L > $M/j1.log 2>&1 &
S="$(echo $B | tr ' ' '+')"; X=""
for e in "d15:drop1:40:1:5" "d210:drop1:40:2:10" "s15:stretch:20:1:5" "s210:stretch:20:2:10" "f120:ficus:20:1:20" "f306:ficus:20:3:6"; do IFS=: read v t fr a b <<< "$e"
  X="$X $v:$t:$fr:$S+--integ+be+--newton+$a+--cg+$b+--tolN+0+--tolP+0"; done
for vv in "fbe:--integ+be" "fhh:--ga+hht+--rho+0.7" "fbeo:--integ+be+--gq_oct+0.1" "fhho:--ga+hht+--rho+0.7+--gq_oct+0.1"; do v=${vv%%:*}; I=${vv#*:}
  X="$X ${v}:ficus:125:$S+$I+--newton+2+--cg+10+--tolN+1e-2+--tolP+1e-1"; X="$X l${v#f}:L000:80:$S+$I+--newton+1+--cg+5+--tolN+1e-2+--tolP+1e-1"; done
mkdir -p /dev/shm/dkta/lbours/fem_an
AN_LIST="$X" bash exe/fem_an_tolx.sh tolx2 > $M/tolx2.log 2>&1
wait
/home/dkta/.conda/envs/af/bin/python exe/fem_an_tol_table.py /dev/shm/dkta/lbours/fem_an/j1 > $M/j1_table.md 2>&1
echo "[j1 끝] $(date '+%F %T')" > $M/j1.END; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
