#!/bin/bash
# 절 CM140: 판 이름 -> 명령줄 (exe/cm139_gen.sh 의 gen_cmd 위에서, 출발점 = CM139 Kacdhnz). source 해서 쓴다 (cm139_gen.sh 먼저).
#   이름 (뒤 _2 _3 은 반복): B = Kacdhnz 그대로 (대조) · A1 = --anx 1 (Anderson 깊이 1) · A2 = --anx 2 (질량 가중)
#   WC / WS / WCS = 출발점에 --w5c 1 / --w5s 1 / 둘 (절 CM70 정의: C = 저질량 고정 노드를 구속 예측으로, S = 구속 받침을 정답 격자에서)
#   ND / L0 / NDL0 = --descent 끔 / --ls_n 0 / 둘 (T 접두와 같이: TND ...)
#   T 접두 = 시간 예산 끔 (TB, TWC, TWS ...)
#   V = 수렴 (뉴턴 60 · PCG 100 · tolN 1e-4 · tolP 1e-3, 시간 예산 끔; 절 CM134 cv 와 같은 바꿈) 뒤에 A1/A2 가 붙으면 V + 그 가속 (VB = 수렴 대조, VA1, VA2)
#   N12 = 예산 판에서 뉴턴 12 · PCG 10 (tolN·tolP 그대로)이 앞에 붙는 조합: N12B · N12A1
cm140_cmd() { local RS=""; local EX=""; local n=${1%%_*}; case $n in *+*) EX=${n#*+}; n=${n%%+*};; esac; local src=$2 out=$3 C X="" S=""
  C=$(gen_cmd Kacdhnz $src $out)
  local TB=""; case $n in T*) TB="s#--tb_w 2 exe/#--tb_w 2 --tb 1000 exe/#"; n=${n#T};; esac   # T = 시간 예산 끔 (--tb 1000; 절 CM134 cv 와 같은 바꿈, 뉴턴·PCG 한도는 그대로 6·5)
  case $n in V*) S="s#--newton 6 --cg 5 --tolN 1e-2 --tolP 1e-1#--newton 60 --cg 100 --tolN 1e-4 --tolP 1e-3#;s#--tb_w 2 exe/#--tb_w 2 --tb 1000 exe/#;s# --xit_from 2##;s# --xg 1##"; n=${n#V};;
             N12*) S="s#--newton 6 --cg 5#--newton 12 --cg 10#"; n=${n#N12};; esac
  case $n in W[CS]*) local W5=""; [[ $n == *C* ]] && W5="${W5}--w5c 1 "; [[ $n == *S* ]] && W5="${W5}--w5s 1 "; C=$(echo "$C" | sed -e "s#--l3fr 1 #--l3fr 1 $W5#"); n="B";; esac   # WC = 절 CM70 C 만 · WS = S_구속 만 · WCS = 둘 (= CM139 F)
  case $n in ND) S="s# --descent##"; n="B";; L0) S="s#--ls_n 10#--ls_n 0#"; n="B";; NDL0) S="s# --descent##;s#--ls_n 10#--ls_n 0#"; n="B";; esac   # ND = 하강 방향 보장 끔 · L0 = 에너지 선탐색 끔 (보폭은 장벽·뒤집힘 실현 가능 한도만)
  case $n in P*) S="${S:+$S;}s#--pbc --mlow#--pbc --pbc_phi --mlow#"; n=${n#P}; [ -z "$n" ] && n=B;; esac   # P 접두 (절 CM141 (b)): --pbc_phi (절 CM27 (e), 정답 서브스텝 구속의 닫힌 꼴 k_구속 = φ(x), 기본 판은 e^-x)
  case $n in F*) RS="s#--frames 50 #--frames 12 --relfix 1.0 --lift 0.5 #"; n=${n#F}; [ -z "$n" ] && n=B;; esac   # F 접두 (절 CM142): 앞면 x=1.0 고정 (x>1.0 만 t=0 해제, 나머지 영구 구속) + 0.5 m 들어올림
  case $n in R*) RS="s#--frames 50 #--frames 12 --rel0 --lift 0.5 #"; n=${n#R}; [ -z "$n" ] && n=B;; esac   # R 접두 (절 CM141): 순차 방출 없이 t=0 에 동시 해제 + 0.5 m 들어올림, 12 프레임 (RB, RV = 수렴, RT...)
  case $n in A1) X="exe/anderson_wrap.py --anx 1 ";; A2) X="exe/anderson_wrap.py --anx 2 ";; B|"") X="";; *) echo "모르는 판 $1" >&2; return 1;; esac
  C=$(echo "$C" | sed -e "s#exe/startpt_wrap.py --sp 1 #exe/startpt_wrap.py --sp 1 $X#")
  [ -n "$S" ] && C=$(echo "$C" | sed -e "$S")
  [ -n "${RS:-}" ] && C=$(echo "$C" | sed -e "$RS")
  [ -n "$TB" ] && C=$(echo "$C" | sed -e "$TB")
  local t; for t in ${EX//+/ }; do case $t in gk) C=$(echo "$C" | sed -e "s#--s9c 1 --s9y 1#--w5gk0 1 --s9c 1 --s9y 1#");; gm) C=$(echo "$C" | sed -e "s#--g4 1e-1 #--g4 1e-1 --g4m 0 #");; es) C=$(echo "$C" | sed -e "s#--tag wolf#--e_scale 0.01 --tag wolf#");; c) C=$(echo "$C" | sed -e "s#--l3fr 1 #--l3fr 1 --w5c 1 #");; s) C=$(echo "$C" | sed -e "s#--l3fr 1 #--l3fr 1 --w5s 1 #");; e) C=$(echo "$C" | sed -e "s#exe/startpt_wrap.py --sp 1 #exe/startpt_wrap.py --sp 1 exe/escwhy_wrap.py #;s#^#ESC_F0=14 ESC_F1=23 #");; f) C=$(echo "$C" | sed -e "s#exe/startpt_wrap.py --sp 1 #exe/startpt_wrap.py --sp 1 exe/escwhy_wrap.py #;s#^#ESC_F0=16 ESC_F1=24 #");; p) C=$(echo "$C" | sed -e "s#--pbc --mlow#--pbc --pbc_phi --mlow#");; *) echo "모르는 토큰 $t" >&2; return 1;; esac; done   # + 토큰 (절 CM142): gk 고스트 강성 0 · gm 고스트 질량 0 · es 영률 x0.01 · c --w5c 1
  echo "$C"; }
