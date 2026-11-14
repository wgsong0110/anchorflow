#!/bin/bash
# 절 CM140: 판 이름 -> 명령줄 (exe/cm139_gen.sh 의 gen_cmd 위에서, 출발점 = CM139 Kacdhnz). source 해서 쓴다 (cm139_gen.sh 먼저).
#   이름 (뒤 _2 _3 은 반복): B = Kacdhnz 그대로 (대조) · A1 = --anx 1 (Anderson 깊이 1) · A2 = --anx 2 (질량 가중)
#   WC / WS / WCS = 출발점에 --w5c 1 / --w5s 1 / 둘 (절 CM70 정의: C = 저질량 고정 노드를 구속 예측으로, S = 구속 받침을 정답 격자에서)
#   T 접두 = 시간 예산 끔 (TB, TWC, TWS ...)
#   V = 수렴 (뉴턴 60 · PCG 100 · tolN 1e-4 · tolP 1e-3, 시간 예산 끔; 절 CM134 cv 와 같은 바꿈) 뒤에 A1/A2 가 붙으면 V + 그 가속 (VB = 수렴 대조, VA1, VA2)
#   N12 = 예산 판에서 뉴턴 12 · PCG 10 (tolN·tolP 그대로)이 앞에 붙는 조합: N12B · N12A1
cm140_cmd() { local n=${1%%_*} src=$2 out=$3 C X="" S=""
  C=$(gen_cmd Kacdhnz $src $out)
  local TB=""; case $n in T*) TB="s#--tb_w 2 exe/#--tb_w 2 --tb 1000 exe/#"; n=${n#T};; esac   # T = 시간 예산 끔 (--tb 1000; 절 CM134 cv 와 같은 바꿈, 뉴턴·PCG 한도는 그대로 6·5)
  case $n in V*) S="s#--newton 6 --cg 5 --tolN 1e-2 --tolP 1e-1#--newton 60 --cg 100 --tolN 1e-4 --tolP 1e-3#;s#--tb_w 2 exe/#--tb_w 2 --tb 1000 exe/#;s# --xit_from 2##;s# --xg 1##"; n=${n#V};;
             N12*) S="s#--newton 6 --cg 5#--newton 12 --cg 10#"; n=${n#N12};; esac
  case $n in W[CS]*) local W5=""; [[ $n == *C* ]] && W5="${W5}--w5c 1 "; [[ $n == *S* ]] && W5="${W5}--w5s 1 "; C=$(echo "$C" | sed -e "s#--l3fr 1 #--l3fr 1 $W5#"); n="B";; esac   # WC = 절 CM70 C 만 · WS = S_구속 만 · WCS = 둘 (= CM139 F)
  case $n in A1) X="exe/anderson_wrap.py --anx 1 ";; A2) X="exe/anderson_wrap.py --anx 2 ";; B|"") X="";; *) echo "모르는 판 $1" >&2; return 1;; esac
  C=$(echo "$C" | sed -e "s#exe/startpt_wrap.py --sp 1 #exe/startpt_wrap.py --sp 1 $X#")
  [ -n "$S" ] && C=$(echo "$C" | sed -e "$S")
  [ -n "$TB" ] && C=$(echo "$C" | sed -e "$TB")
  echo "$C"; }
