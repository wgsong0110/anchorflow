#!/bin/bash
# 명시적 FEM 첫 확인 (절 L): 기준·최적화 x lag·sub 를 짧게 (6 프레임) -- 컴파일·값 대조. 사용 (잡 안에서만): AF_DETACH=1 bash exe/job_run.sh <작업트리>/exe/fem_ex_smoke.sh <로그>
cd "$(dirname "$0")/.." || exit 1
S=${S:-smoke}; NN=${NN:-64}; FR=${FR:-6}
for md in lag sub; do
  bash exe/fem_ex_run.sh ${S}_${md}_base_n$NN 0 --mode $md --N $NN --frames $FR --no_dump --warm 1
  bash exe/fem_ex_run.sh ${S}_${md}_opt_n$NN 0 --mode $md --N $NN --frames $FR --no_dump --warm 1 --opt
done
echo SMOKE_DONE
