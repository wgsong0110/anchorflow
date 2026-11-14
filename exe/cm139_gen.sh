#!/bin/bash
# 절 CM139: 판 이름 (요소 글자 모음) -> 명령줄. source 해서 쓴다. 기준 명령 = R2 cm131/k5 (H5 + --w8k 1) 에서 기록 감싸개 (resmode·escwhy·ESC_*·RM_*) 만 뺀 것.
#   글자 (H5 -> H9 차이 요소; K 는 k5 에 이미 있음):
#     K  --w8k 1 (고스트 P1X 강성; 없으면 --w8k 빠짐)        a  env AF_E1_DG=1 AF_G1_DG=1 (진단)
#     n  speed13_wrap 대신 nlwhy_wrap (기록)                  c  wolfh12_wrap --w12 0 --w12dg 1
#     d  finterp_wrap --fi 0 --fidg 1                         h  wolfh11_wrap + wolfh9_wrap (인자 없음 = 모듈 안 부름)
#     z  speed12_wrap --s12z 1                                E  확장 --w8e 1 --w8d 2 --w8t 1
#     F  --w5s 1 --w5c 1 (S_구속 + C)
#   k5 = "K", H9 = "KacdhnzEF". 예: KEF = k5 + 확장 + S_구속·C, KacdhnzF = H9 에서 확장만 뺌.
#   gen_cmd <이름> <k5 명령 파일> <출력 디렉토리> -> 표준출력에 명령줄
gen_cmd() { local n=${1%%_*} src=$2 out=$3 C PRE CH="" W8="" W5="" X13="exe/speed13_wrap.py " S12="exe/speed12_wrap.py "
  C=$(tr -d "\n" < $src)
  C=$(echo "$C" | sed -e 's#^ESC_F0=5 ESC_F1=30 RM_F0=5 RM_F1=30 ##' -e 's#exe/resmode_wrap.py exe/escwhy_wrap.py ##')
  PRE="AF_SP_FR=1,50"; [[ $n == *a* ]] && PRE="$PRE AF_E1_DG=1 AF_G1_DG=1"
  [[ $n == *n* ]] && X13="exe/nlwhy_wrap.py "
  [[ $n == *c* ]] && CH="${CH}exe/wolfh12_wrap.py --w12 0 --w12dg 1 "
  [[ $n == *d* ]] && CH="${CH}exe/finterp_wrap.py --fi 0 --fidg 1 "
  [[ $n == *h* ]] && CH="${CH}exe/wolfh11_wrap.py exe/wolfh9_wrap.py "
  [[ $n == *z* ]] && S12="exe/speed12_wrap.py --s12z 1 "
  [[ $n == *K* ]] && W8="--w8k 1 "; [[ $n == *E* ]] && W8="${W8}--w8e 1 --w8d 2 --w8t 1 "
  [[ $n == *F* ]] && W5="--w5s 1 --w5c 1 "
  local REST=${C#*--l3fr 1 --s9c 1}; local PY=${C%% -u *}; PY=${PY#AF_SP_FR=1,50 }
  echo "$PRE $PY -u exe/startpt_wrap.py --sp 1 ${X13}${CH}${S12}exe/ext2_wrap.py ${W8}--l4 1 --l3 10 --l3sc 1 --l3fr 1 ${W5}--s9c 1$REST" | sed -e "s#--out [^ ]*\$#--out $out#"; }
