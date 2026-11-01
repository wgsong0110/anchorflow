#!/bin/bash
# 후보 0 (진단, 비측정): B-스플라인 요소 torch 판 (절 AK.20) 을 θ 상한 없이 ficus 에 -> 잎 끝 주기 (θ 0.5 판 ficus_b05 와 함께).
cd "$(dirname "$0")/.." || exit 1
BS_V=bn BS_X="--theta 0" BS_TS=ficus bash exe/fem_bsel_run.sh
bash exe/fem_ak_period.sh fem_ak_bsel/ficus_bn fem_ak_bsel/ficus_b05
grep -a "" /tmp/dkta/lbours/fem_ak_bsel/ficus_bn/log.txt | tail -2 | cut -c1-200
