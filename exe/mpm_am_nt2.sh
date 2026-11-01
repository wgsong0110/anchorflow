#!/bin/bash
# 절 AM 2 차 비측정: 루프 완전 펼침 (가중치 배열이 국소 메모리에 있던 것 없앰) 뒤 주요 판 다시
cd "$(dirname "$0")/.." || exit 1
until [ -f /tmp/dkta/lbours/mpm_speed/am_nt/AM_DONE ]; do sleep 60; done   # 1 차 묶음과 카드가 겹치지 않게
AM_TAG=am_nt2 AM_VARS="c8_shared c5_graph c0_port c1_sort c4_wred c8_shared_nowred c6_nosort c8_shared_r2 floor" bash exe/mpm_am_run.sh
