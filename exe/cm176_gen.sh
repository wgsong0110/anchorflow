#!/bin/bash
# 절 CM176: 판 이름 -> 명령줄 (cm139·cm140·cm170·cm174_gen.sh 먼저 source). 이름 맨 끝 토큰 (하나만):
#   +ss = --ssn 1 (반매끈 뉴턴: 정확한 활성 집합 접선 --ctan 2 + 반환 뒤 잔차 --rls 2 비단조 Armijo 기억 4; 에너지 선탐색·하강 방향 끔 = rl 토큰과 같은 바꿈)
#   +sm = +ss 에서 기억 1 (단조 Armijo; AF_RLS_NM=1) -- 비단조 효과 가르기용
# 예: cm176_cmd TFB+r+ss src out
cm176_cmd() { local n=${1%%_*} SS=""
  case $n in *+ss) SS=4; n=${n%+ss};; *+sm) SS=1; n=${n%+sm};; esac
  local C; C=$(cm174_cmd $n $2 $3) || return 1
  if [ -n "$SS" ]; then
    C=$(echo "$C" | sed -e "s# --ls_n 10##;s# --en_mo##;s# --descent##;s#exe/startpt_wrap.py --sp 1 #exe/startpt_wrap.py --sp 1 exe/ssn_wrap.py --ssn 1 #;s#^#AF_RLS_NM=$SS #")
  fi
  echo "$C"; }
