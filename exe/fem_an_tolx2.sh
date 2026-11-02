#!/bin/bash
# 절 AN.14 단독 시간 tolx2 (카드 0): 낙하·블록 신축 상한 1x5·2x10, ficus 상한 모양 1x20·3x6 (허용치 0 = 최악 시간), ficus 2x10 커널별 시간 (--prof 5).
cd "$(dirname "$0")/.." || exit 1
export PATH=$HOME/bin:$PATH
B="--cell+1+--mg_levels+4+--mg_nc+4+--ls_n+10+--descent+--K+2+--mg_wreuse+--mg_lreuse+--integ+be"
X=""
for e in "d15:drop1:20:1:5" "d210:drop1:20:2:10" "s15:stretch:12:1:5" "s210:stretch:12:2:10" "f120:ficus:20:1:20" "f306:ficus:20:3:6" "f210:ficus:20:2:10"; do IFS=: read v t fr a b <<< "$e"
  X="$X $v:$t:$fr:$B+--newton+$a+--cg+$b+--tolN+0+--tolP+0"; done
X="$X fprof:ficus:12:$B+--newton+2+--cg+10+--tolN+0+--tolP+0+--prof+5 lprof:L000:12:$B+--newton+1+--cg+5+--tolN+0+--tolP+0+--prof+5"
AN_LIST="$X" bash exe/fem_an_tolx.sh tolx2
