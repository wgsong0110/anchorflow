#!/bin/bash
# 절 AM 비측정 전부 (80 프레임, NOT_TIMED): 구간 비율·개선 배수·결과 불변
cd "$(dirname "$0")/.." || exit 1
AM_TAG=am_nt bash exe/mpm_am_run.sh
