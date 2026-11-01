#!/bin/bash
# 절 AM 3 차 비측정: 2 차 묶음 (GPU0, 카드 자체는 비어 있었음) 에 없던 판 -- 원래 솔버 pg 를 같은 조건으로, 융합·희소·워프 합 없음, 그래프 재현
cd "$(dirname "$0")/.." || exit 1
AM_TAG=am_nt3 AM_VARS="pg c5_graph_r2 c2_fuse c3_sparse c7_nowred" bash exe/mpm_am_run.sh
