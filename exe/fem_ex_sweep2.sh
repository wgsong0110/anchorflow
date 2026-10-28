#!/bin/bash
# 절 L 스윕 2: 노드 질량 사면체 등분 (--mass tet, 정의 변경) -- lag/sub 최적화 N 스윕 (lego L000 80 프레임), 바닥 없음 경계 (20 프레임)
# 앞 스윕이 끝날 때까지 기다린다 (단독 측정 순서 보장): WAIT_FOR=<로그> 에 *_DONE 이 나올 때까지
cd "$(dirname "$0")/.." || exit 1
[ -n "$WAIT_FOR" ] && while ! grep -q "_DONE" "$WAIT_FOR" 2>/dev/null; do sleep 30; done
for NN in ${NLAG:-24 32 48 64 96 128 256}; do
  bash exe/fem_ex_run.sh s2_lag_tm_n$NN 0 --mode lag --N $NN --opt --near32 --mass tet --no_dump
done
for NN in 24 32 48 64; do
  bash exe/fem_ex_run.sh s2_lag_tm_nofl_n$NN 0 --mode lag --N $NN --opt --near32 --mass tet --floor none --frames 20 --no_dump
done
for NN in ${NSUB:-32 48 64 128 256}; do
  bash exe/fem_ex_run.sh s2_sub_tm_n$NN 0 --mode sub --N $NN --opt --near32 --mass tet --no_dump
done
echo SWEEP2_DONE
