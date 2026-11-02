#!/bin/bash
# 절 AN.err 2 차 본 묶음 (조정자 11-02; 비측정, 정확도만): 빈 카드 하나에 하나씩, 출력 /dev/shm -> R2 lbours/fem_anerr/r2/<판>_<궤적>
# (1) θ 0.55·0.6·TR-BDF2 + 관성 질량 하한 (절 AH.18 κ 0.1 + 절 AG.15 국소 CFL 명시적 N 267 기준) x K 1·2
# (2) 적응 걸음 (--kad 4·8, 판정은 프레임 시작 입자 위치·속도만) (3) 고착 인자 걸음 안 평균 (--fl_avg 8) -- 둘 다 θ 0.55 + 하한 K 1 위
# (4) 빠진 lego 후퇴 오일러 K 2·4·8 (8 궤적)
# 씬: lego 0001·0003·0016·0018 (실시간 판 미달 상위 4) + 대조 0005·0013 + L000 + ficus (주기) + 낙하 블록
cd "$(dirname "$0")/.." || exit 1
mkdir -p /dev/shm/dkta; L=/dev/shm/dkta/anerr_r2.txt; : > $L
S="--pc mg --newton 6 --cg 30 --ls_diff --en"; M="--pc mg --newton 4 --cg 20 --ls_diff --en"
FL="--mlow 0.1 --mfl_n 267"
SC="drop1 0001 0003 0016 0018 0005 0013 L000 ficus"
for k in 1 2; do for t in $SC; do
  echo "t55_K$k $t --K $k --integ theta --theta 0.55 $FL $S" >> $L
  echo "t60_K$k $t --K $k --integ theta --theta 0.6 $FL $S" >> $L
  echo "trb_K$k $t --K $k --integ trbdf2 $FL $S" >> $L
done; done
for t in $SC; do
  echo "t55ad4_K1 $t --K 1 --kad 4 --integ theta --theta 0.55 $FL $S" >> $L
  echo "t55ad8_K1 $t --K 1 --kad 8 --integ theta --theta 0.55 $FL $M" >> $L
  echo "t55av_K1 $t --K 1 --fl_avg 8 --integ theta --theta 0.55 $FL $S" >> $L
done
for t in 0001 0003 0016 0017 0018 0005 0013 L000; do
  echo "g_be_K2 $t --K 2 --integ be $S" >> $L
  echo "g_be_K4 $t --K 4 --integ be $S" >> $L
  echo "g_be_K8 $t --K 8 --integ be $M" >> $L
done
wc -l $L
ERR_PAR=6 bash exe/fem_anerr_pool.sh r2 $L
