#!/bin/bash
# 절 AM 시험 (3 프레임, 비측정): 컴파일·식 확인용
cd "$(dirname "$0")/.." || exit 1
AM_TAG=am_smoke AM_EXTRA="--frames 3 --prof 2" AM_VARS="c5_graph c0_port c2_fuse pg" bash exe/mpm_am_run.sh
