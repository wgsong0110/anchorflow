#!/bin/bash
# 절 CM178 묶음 tetfix: 판 이름 -> 명령줄. 바탕 = R2 lbours/tetc 의 레고 판 명령 그대로 (cL_1 = TC 영상 판 · dL_1 = TC 진단 판), 바뀐 것만 sed 로.
#   사용: tf_cmd <이름> <바탕 cL1 파일> <바탕 dL1 파일> <out> <oXL1 파일 (H9 육면체)>   (출력 = 명령 한 줄)
#   이름: TR = dL_1 (--tcdg 2 --hgt --pdiag --qjd --bar_cnt) + --nrec + tetfix_wrap --tfx 1 (창 6:13), 14 프레임 (추적, 결과 불변)
#         TC = cL_1 그대로 (TC 영상 판, 80 프레임) · H9 = oXL_1 (육면체 기본 판, 80 프레임)
#         +ff·+ff1 = --tff 1 (선탐색 뒤집힘 불가능) · +ff2 = --tff 2 (시작점 섞기에 사면체 점검, --y0loc 3 -> 6) · +ff3 = 둘 다
#         뒤 토큰: +f0:f1 프레임 수 지정 (+n20 = 20 프레임) · +x<문자열> = 수정 플래그 (원인 확정 뒤 정의)
tf_cmd() { local n=$1 C1=$2 D1=$3 O=$4 H9=$5 base fr="" extra=""
  local nm=${n%%+*}; local rest=${n#"$nm"}; rest=${rest#+}
  local ff=0; for tok in ${rest//+/ }; do case $tok in n[0-9]*) fr=${tok#n};; dump) extra="$extra --dump --dump_F --dump_part";; ff) ff=1;; ff[123]) ff=${tok#ff};; x*) extra="$extra ${tok#x}";; esac; done
  local sc=L; case $nm in TCW*|H9W*) sc=W;; TCF*|H9F*) sc=F;; TCB*|H9B*) sc=B;; esac
  case $nm in
    TCW*|TCF*|TCB*) base=$(cat ${C1%cL1.txt}c${sc}1.txt); norr=0;;
    H9W*|H9F*|H9B*) base=$(cat ${H9%oXL1.txt}oX${sc}1.txt); norr=0;;
    TR*) base=$(cat $D1); 
         base=${base/exe\/tetc_wrap.py/exe\/tetfix_wrap.py --tfx 1 --tfw 6:13 exe\/tetc_wrap.py}
         base=${base/--hgt --pdiag/--nrec --hgt --pdiag}; [ -z "$fr" ] && fr=14;;
    TC*) base=$(cat $C1);;
    H9*) base=$(cat $H9);;
    *) echo "모르는 판 $n" >&2; return 1;;
  esac
  if [ "${norr:-0}" = 1 ]; then base=$(echo "$base" | sed -e "s#--rr pg --rr_shape [^ ]* --rr_traj [^ ]* --rr_cfg [^ ]* --rr_fast 1##;s#--tc 8##;s#--tb 16.7 --tb_m 0.8 --tb_w 2##"); fi
  if [ $ff -ge 1 ]; then
    case $base in *tetfix_wrap.py*) base=${base/tetfix_wrap.py/tetfix_wrap.py --tff $ff};; *) base=${base/exe\/tetc_wrap.py/exe\/tetfix_wrap.py --tfx 0 --tff $ff exe\/tetc_wrap.py};; esac
  fi
  [ $ff -ge 2 ] && [ $((ff & 2)) = 2 ] && base=${base//--y0loc 3/--y0loc 6}
  base=$(echo "$base" | sed -e "s#--out [^ ]*#--out $O#")
  [ -n "$fr" ] && base=$(echo "$base" | sed -e "s#--frames [0-9]*#--frames $fr#")
  echo "$base $extra"; }
