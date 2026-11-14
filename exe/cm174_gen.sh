#!/bin/bash
# 절 CM174: 판 이름 -> 명령줄 (exe/cm139_gen.sh · cm140_gen.sh · cm170_gen.sh 먼저 source). 이름 뒤 +ed = 이탈 추적 기록 (결과 불변): escwhy (걸음별 노드·가우스 점, ESC_F0~F1) + cm174 (뉴턴 반복별 노드 힘 탄성·꼬리 분해·점 상태, CM174_F0~F1).
#   프레임 창 기본 12~17 (env CM174_F0·F1). 예 cm174_cmd TB+rl+ct1+w2+ed src out
cm174_cmd() { local n=${1%%_*} ED=""
  case $n in *+ed) ED=1; n=${n%+ed};; esac
  local C; C=$(cm170_cmd $n $2 $3) || return 1
  if [ -n "$ED" ]; then
    local F0=${CM174_F0:-12} F1=${CM174_F1:-17}
    C=$(echo "$C" | sed -e "s#exe/startpt_wrap.py --sp 1 #exe/startpt_wrap.py --sp 1 exe/escwhy_wrap.py #;s#exe/startpt_wrap.py#exe/cm174_wrap.py exe/startpt_wrap.py#;s#^#ESC_F0=$F0 ESC_F1=$F1 CM174_F0=$F0 CM174_F1=$F1 #")
  fi
  echo "$C"; }
