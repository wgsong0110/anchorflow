#!/bin/bash
# 절 AM 단독 측정: 측정 카드에 다른 계산 프로세스만 없으면 된다 (11-02 정정). AM_GPU 를 주지 않으면 카드 0 에서 6 중 계산 프로세스가 없고 500 MiB 미만인 첫 카드.
# AM_VARS 기본 전부. 결과 R2 lbours/mpm_speed/${AM_TAG:-am_timed}
cd "$(dirname "$0")/.." || exit 1
if [ -z "${AM_GPU:-}" ]; then
  for g in ${AM_CANDS:-0 1 2 3 4 5 6}; do
    u=$(nvidia-smi -i $g --query-gpu=uuid,memory.used --format=csv,noheader,nounits)
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "${u%%,*}")
    [ "$n" -eq 0 ] && [ "${u##*, }" -lt 500 ] && { AM_GPU=$g; break; }
  done
fi
[ -z "${AM_GPU:-}" ] && { echo "빈 카드 없음 $(date +%T)"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; exit 1; }
echo "[측정 카드] $AM_GPU $(date +%T)"
AM_TIMED=1 AM_GPU=$AM_GPU AM_TAG=${AM_TAG:-am_timed} bash exe/mpm_am_run.sh
