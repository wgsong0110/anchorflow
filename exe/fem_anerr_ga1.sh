#!/bin/bash
# 절 AN.err3 1 단계 (조정자 11-02: generalized-α·HHT-α): 첫 시험 (3 프레임) -> 단위 시험 (ficus 주기, 블록 신축·전단 40 프레임, 낙하 반발) x 판. 카드 3·5 만, 출력 /dev/shm -> R2 lbours/fem_anerr/ga0·ga1
cd "$(dirname "$0")/.." || exit 1
mkdir -p /dev/shm/dkta
S="--pc mg --newton 6 --cg 30 --ls_diff --en"
L0=/dev/shm/dkta/anerr_ga0.txt
cat > $L0 <<EOT
s_ga7 drop1 --K 1 --integ ga --rho 0.7 $S --frames 4
s_hht7 stretch --K 1 --integ hht --rho 0.7 $S --frames 3
s_ga5 ficus --K 2 --integ ga --rho 0.5 $S --frames 3
EOT
ERR_CARDS="3 5" ERR_PAR=2 bash exe/fem_anerr_pool.sh ga0 $L0
if grep -q "rror\|발산\|폭주" /dev/shm/dkta/lbours/fem_anerr/ga0/done.txt || [ $(grep -c 롤아웃 /dev/shm/dkta/lbours/fem_anerr/ga0/done.txt) -lt 3 ]; then echo "[첫 시험 실패] 1 단계 안 띄움"; cat /dev/shm/dkta/lbours/fem_anerr/ga0/done.txt; exit 1; fi
L=/dev/shm/dkta/anerr_ga1.txt; : > $L
for t in ficus stretch shear drop1; do
  for k in 1 2; do
    for r in 0.5 0.7 0.9; do echo "ga${r#0.}_K$k $t --K $k --integ ga --rho $r $S" >> $L; done
    for r in 0.7 0.9; do echo "hht${r#0.}_K$k $t --K $k --integ hht --rho $r $S" >> $L; done
  done
  echo "be_K2 $t --K 2 --integ be $S" >> $L
  echo "mid_K2 $t --K 2 --integ mid $S" >> $L
done
sed -i 's/^\(.* \(stretch\|shear\) .*\)$/\1 --frames 40/' $L
wc -l $L
ERR_CARDS="3 5" ERR_PAR=2 bash exe/fem_anerr_pool.sh ga1 $L
