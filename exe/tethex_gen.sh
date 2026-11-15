#!/bin/bash
# 절 CM180 묶음 tethex: 혼합 판 (탄성 사면체 + 입자<->노드 전달 이차 B-스플라인, tet1_wrap --tmx 1) 레고 명령. 바탕 = R2 lbours/tetfix/TC+ff3/cmd.txt 그대로, 바뀐 것만:
#   tetc_wrap (--tcm 1 --tcdg 1: 사면체 질량·운동량 모으기 + 노드 1/4) 뺌, --gq_mass 뺌 (P2G 는 rt_p2g 의 B-스플라인 노드 몫), tet1_wrap 에 --tmx 1. --gq_oct 0 (κ 0)·--tff 3·--y0loc 6 그대로.
#   사용: th_cmd <TC+ff3 cmd 파일> <out> [tmx: 1 = P2G·G2P 둘 다 B-스플라인 (기본), 2 = P2G 만] [tmx 1|2]
th_cmd() { local b tm=${3:-1}; b=$(cat "$1")
  b=${b/exe\/tetc_wrap.py --tcm 1 --tcdg 1/}
  b=${b/exe\/tet1_wrap.py --tet 1/exe\/tet1_wrap.py --tet 1 --tmx $tm}
  b=${b/--gq_mass/}
  echo "$b" | sed -e "s#--out [^ ]*#--out $2#"; }
