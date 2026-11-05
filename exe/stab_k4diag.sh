#!/bin/bash
# 절 S.8 (1): 우리 (B) lego L000 '비단조: K 2 안정, K 4 발산' 원인 진단 (잡/인스턴스 안, 커밋 디렉토리에서; 카드 $1, 출력 $2).
# 모든 판 결정적 (--det), 기준 = exe/stab.py ours_cmd 와 같은 명령 (cm_scene L000 --v=b + --ga hht --rho 0.8 --K k --newton 2 --cg 4 --tolN 1e-2 --tolP 1e-1).
#   재현: k4a·k4b (같은 명령 두 번 -> 발산 프레임·끝 해시 같아야), k4d = + 진단 (--diag --pdiag --bar_cnt --hgt, 결과 불변 -> k4a 와 해시 같아야), k1d·k2d 비교.
#   결과 불변이라 적힌 손잡이 끔 (해시가 바뀌면 정의 불변 결함 후보): nolsf (--lsfast), nofk (--fk), nohs (--hspec --ncomp), nomv2 (--mv2 --hspec --ncomp).
#   정의 바꾸는 손잡이 (진단 전용, 후보): noeb (--y0eb 끔), nowr (--mg_wreuse 끔), nolr (--mg_lreuse 끔), it4 (뉴턴 4·PCG 8), it8 (뉴턴 8·PCG 16·εN 1e-3·εP 1e-2),
#     ls20 (선탐색 후보 20), nodesc (--descent 끔), r05 (HHT ρ 0.5), va (바닥 = (A) 가우스 점 고착, 장벽 없음), dh2 (장벽 두께 2 배), bst2 (띠 고착 정답 가중).
#   K 지도: K 3·5·6·8·12·16 (기준 명령).
# 표: exe/stab_k4diag_tab.py -> $2/k4diag_table.md. 판 결과 (summary.json) 있으면 건너뜀 (이어 돌리기).
G=$1; O=$2; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export CUDA_VISIBLE_DEVICES=$G PYTHONUTF8=1 OMP_NUM_THREADS=8
base() {   # base K -> 명령 (SIM DA XA)
  bash -c "source exe/fem_cm8_scn.sh >/dev/null 2>&1; cm_scene L000 $2 --ga hht --rho 0.8 --K $1 --newton 2 --cg 4 --tolN 1e-2 --tolP 1e-1 --det >/dev/null 2>&1; echo \"\$SIM \$DA \$XA\""; }
run() {    # run 이름 K 변형(--v=b|'') '뺄 인자' '더할 인자'
  local n=$1 k=$2 v=$3 drop=$4 add=$5 c
  [ -f $O/$n/summary.json ] && return
  c=" $(base $k "$v") "
  for x in $drop; do c=${c// $x / }; done
  mkdir -p $O/$n; echo "$PY -u $c --tag L000 --out $O/$n $add" > $O/$n/cmd.txt
  echo "[$(date '+%T')] $n" >> $O/k4diag.log
  $PY -u $c --tag L000 --out $O/$n $add > $O/$n/log.txt 2>&1
  echo "[$(date '+%T')] $n rc=$? $(grep -h '롤아웃\]' $O/$n/log.txt | tail -1)" >> $O/k4diag.log
}
D="--diag --pdiag --bar_cnt --hgt"
run k4a 4 --v=b "" ""
run k4b 4 --v=b "" ""
run k4d 4 --v=b "" "$D"
run k2d 2 --v=b "" "$D"
run k1d 1 --v=b "" "$D"
run nolsf 4 --v=b "--lsfast" "$D"
run nofk 4 --v=b "--fk" "$D"
run nohs 4 --v=b "--hspec --ncomp" "$D"
run nomv2 4 --v=b "--mv2 --hspec --ncomp" "$D"
run noeb 4 "" "" "$D"
run nowr 4 --v=b "--mg_wreuse" "$D"
run nolr 4 --v=b "--mg_lreuse" "$D"
run it4 4 --v=b "" "$D --newton 4 --cg 8"
run it8 4 --v=b "" "$D --newton 8 --cg 16 --tolN 1e-3 --tolP 1e-2"
run ls20 4 --v=b "" "$D --ls_n 20"
run nodesc 4 --v=b "--descent" "$D"
run r05 4 --v=b "" "$D --rho 0.5"
run va 4 --v=a "" "$D"
run dh2 4 --v=b "" "$D --fl_dhm 2"
run bst2 4 --v=b "" "$D --fl_bst 2"
for k in 3 5 6 8 12 16; do run K$k $k --v=b "" ""; done
$PY exe/stab_k4diag_tab.py $O > $O/k4diag_table.md 2>&1
echo "[$(date '+%T')] K4DIAG_END" >> $O/k4diag.log
