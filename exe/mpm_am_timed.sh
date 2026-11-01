#!/bin/bash
# 절 AM 단독 측정 (다음 잡에서, 다른 단독 측정과 함께): 판마다 카드 1~6 이 10 MiB 이하가 될 때까지 기다린 뒤 gpu0.lock + GPU0.
# 결과 불변은 비측정 판 (am_nt) 에서 확인했으므로 덤프는 원래 솔버 판·최종 판만 남긴다 (나머지도 남겨도 시간에는 영향 없음 -- 내보내기는 시간 밖).
cd "$(dirname "$0")/.." || exit 1
AM_TIMED=1 AM_TAG=am_timed bash exe/mpm_am_run.sh
