#!/bin/bash
# 절 L 진단 2 (프레임 단위 바닥 에너지): 첫 프레임 노드 국소 안정 걸음 + 서브스텝 |v|, 짧은 롤아웃으로 발산 경계 (기준 lag), 최적화 판 동작 확인
cd "$(dirname "$0")/.." || exit 1
S=${S:-d2}
for NN in 1024 2048; do
  bash exe/fem_ex_run.sh ${S}_lag_n$NN 0 --mode lag --N $NN --frames 10 --no_dump --warm 1 --diag
done
for NN in 1024 2048; do
  bash exe/fem_ex_run.sh ${S}_lag_opt_n$NN 0 --mode lag --N $NN --frames 10 --no_dump --warm 1 --opt --dump_pred
  bash exe/fem_ex_run.sh ${S}_sub_opt_n$NN 0 --mode sub --N $NN --frames 10 --no_dump --warm 1 --opt
done
bash exe/fem_ex_run.sh ${S}_lag_n2048_p 0 --mode lag --N 2048 --frames 10 --no_dump --warm 1 --dump_pred
bash exe/fem_ex_run.sh ${S}_sub_n1024 0 --mode sub --N 1024 --frames 3 --no_dump --warm 1
echo DIAG_DONE
