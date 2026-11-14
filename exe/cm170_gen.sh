#!/bin/bash
# 절 CM170: 판 이름 -> 명령줄 (exe/cm139_gen.sh · exe/cm140_gen.sh 먼저 source). 이름 뒤 +r = 먼 쪽 평형 잔차 기록 (exe/cm170_wrap.py, 결과 불변; 프레임 창 CM170_F0~F1 기본 1~3).
#   cm170_cmd TFB+r src out   (기본 판, 뉴턴 6 -- cm140 의 TFB 와 같음)   cm170_cmd TVFB+r src out   (뉴턴 60 · PCG 100 · tolN 1e-4, xit_from·xg 뺌)
cm170_cmd() { local n=${1%%_*} R=0 FR="" NG="" PL="" W2=""; local ch=1
  while [ $ch = 1 ]; do ch=0; case $n in *+pl) PL=1; n=${n%+pl}; ch=1;; *+ng) NG=1; n=${n%+ng}; ch=1;; *+w2) W2=1; n=${n%+w2}; ch=1;; *+n4) FR=1; n=${n%+n4}; ch=1;; *+r) R=1; n=${n%+r}; ch=1;; esac; done
  local C; C=$(cm140_cmd $n $2 $3) || return 1
  [ $R = 1 ] && C=$(echo "$C" | sed -e "s#exe/startpt_wrap.py#exe/cm170_wrap.py exe/startpt_wrap.py#;s#^#CM170_F0=${CM170_F0:-1} CM170_F1=${CM170_F1:-3} #")
  [ -n "$PL" ] && C=$(echo "$C" | sed -e "s#exe/startpt_wrap.py#exe/cm167_wrap.py exe/startpt_wrap.py#;s#^#CM167_F0=${CM170_F0:-1} CM167_F1=${CM170_F1:-3} CM167_ZC=-10 #")   # +pl = 점 상태 바뀜 수 기록도 (exe/cm167_wrap.py, 전체 점)
  [ -n "$W2" ] && C=$(echo "$C" | sed -e "s#--rls 1#--rls 2#")   # +w2 = 절 CM173 --rls 2 (관성 계량 잔차 노름; rl 토큰 위에서만)
  [ -n "$NG" ] && C=$(echo "$C" | sed -e "s# --xit_from 2##;s# --xg 1##")   # +ng = 뉴턴 중간 멈춤 (xit_from) 과 나눈 그래프 (xg) 끔: 한도 안의 반복을 다 돎
  [ -n "$FR" ] && C=$(echo "$C" | sed -e "s#--frames 12 #--frames 4 #")   # +n4 = 앞 4 프레임만 (뉴턴 60 판 시간 줄임)
  echo "$C"; }
