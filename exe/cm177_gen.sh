#!/bin/bash
# 절 CM177: 판 이름 -> 명령줄 (cm139·cm140·cm170·cm174·cm176_gen.sh 먼저 source). 서브스텝 증가 (K>1) 허용 (사용자 11-15).
#   이름 = cm176_cmd 이름 그대로: TB+k4 (기본 조합 + K 4) · TB+k4+rl+ct1+w2 (무른 헤시안 ct1 + 잔차 계량 rls 2) · TB+k4+ss (반매끈 --ssn 1)
#   k2·k4·k8 토큰 = --K n 이고 --ksplit 3 --ks_big 뺌 (cm140_gen.sh). 걸음당 뉴턴 6 · PCG 5 는 그대로.
#   맨 끝 +it = 프레임당 총 반복을 K=1 과 맞춤: 걸음당 뉴턴 = ceil(6/K) (K2 3, K4 2, K8 1), xit_from·xg 끔.
#   +k16 · +k32 · +k64 = --K 16 · 32 · 64 (cm140 의 k2 토큰을 받아 K 만 바꿈; 이름 어디에 있어도 됨)
cm177_cmd() { local n=${1%%_*} IT="" KB=""
  case $n in *+it) IT=1; n=${n%+it};; esac
  case $n in *+k16*) KB=16;; *+k32*) KB=32;; *+k64*) KB=64;; esac
  [ -n "$KB" ] && n=${n/+k$KB/+k2}
  local C; C=$(cm176_cmd $n $2 $3) || return 1
  [ -n "$KB" ] && C=$(echo "$C" | sed -e "s#--K 2 #--K $KB #")
  if [ -n "$IT" ]; then
    local K; K=$(echo "$C" | grep -o -- '--K [0-9]*' | head -1 | cut -d' ' -f2); local N=$(( (6 + K - 1) / K ))
    C=$(echo "$C" | sed -e "s#--newton 6 #--newton $N #;s# --xit_from 2##;s# --xg 1##")
  fi
  echo "$C"; }
