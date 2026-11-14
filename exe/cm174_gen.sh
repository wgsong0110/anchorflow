#!/bin/bash
# 절 CM174·CM175: 판 이름 -> 명령줄 (exe/cm139_gen.sh · cm140_gen.sh · cm170_gen.sh 먼저 source). 이름 뒤 토큰 (여러 개 가능):
#   +ed = 이탈 추적 기록 (escwhy + cm174, 창 CM174_F0~F1 기본 12~17)   +ee = 같은 기록, 창 9~13 (절 CM175)
#   +c3 = --ctan 3 (절 CM175: 꼭짓점 점 접선을 체적 압축 탄성 체적 강성만)
cm174_cmd() { local n=${1%%_*} ED="" F0=${CM174_F0:-12} F1=${CM174_F1:-17} C3="" ch=1
  while [ $ch = 1 ]; do ch=0; case $n in *+ed) ED=1; n=${n%+ed}; ch=1;; *+ee) ED=1; F0=9; F1=13; n=${n%+ee}; ch=1;; *+c3) C3=1; n=${n%+c3}; ch=1;; esac; done
  local C; C=$(cm170_cmd $n $2 $3) || return 1
  [ -n "$C3" ] && C=$(echo "$C" | sed -e "s#--ctan 1#--ctan 3#")
  if [ -n "$ED" ]; then
    C=$(echo "$C" | sed -e "s#exe/startpt_wrap.py --sp 1 #exe/startpt_wrap.py --sp 1 exe/escwhy_wrap.py #;s#exe/startpt_wrap.py#exe/cm174_wrap.py exe/startpt_wrap.py#;s#^#ESC_F0=$F0 ESC_F1=$F1 CM174_F0=$F0 CM174_F1=$F1 #")
  fi
  echo "$C"; }
