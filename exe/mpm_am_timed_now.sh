#!/bin/bash
# 절 AM 단독 측정 (조정자 11-02): 지금 빈 카드 4·5·6 중 하나에서 원래 솔버 (pg) 와 최선 판 (c5_graph)
cd "$(dirname "$0")/.." || exit 1
AM_CANDS="4 5 6" AM_TAG=am_timed1 AM_VARS="pg c5_graph" bash exe/mpm_am_timed.sh
