#!/bin/bash
# [CM192] 대기열 작업: 추가 지표 (exe/stab_extra.py) -- pyiqa 없으면 설치 (인스턴스 안). bash exe/sfq_extra.sh <장면> <우리 접미>
python -c "import pyiqa" 2>/dev/null || pip install -q pyiqa "torch==2.5.1" "torchvision==0.20.1" "numpy<2" 2>&1 | tail -2
python -u exe/stab_extra.py --scene $1 --hextag $2
