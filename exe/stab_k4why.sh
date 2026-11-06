#!/bin/bash
# 절 S.9: 우리 (B) lego L000 K 4 발산 -- 몸통 안 가우스 점이 '왜' 뒤집히나 (잡/인스턴스 안, 커밋 디렉토리에서; 출력 $1).
# 기준 명령 = exe/stab_k4diag.sh base() 와 같음 (cm_scene L000 --v=b + --ga hht --rho 0.8 --K k --newton 2 --cg 4 --tolN 1e-2 --tolP 1e-1 --det).
# 판: k4 (K 4 기준), k2 (K 2 완주), hs (K 4 --hspec --ncomp 끔 = 반올림 순서만 다름, 완주), mv (K 4 --mv2 --hspec --ncomp 끔, 완주).
#   <판>w = 진단 켬 (--wdiag 40 --wdiag_end 68 --gfin --diag --pdiag --fhash --dump --dump_part), <판>o = 진단 끔 (--fhash 만: 호스트 해시) -> 프레임별 해시가 같아야 결과 불변.
# 판 결과 (summary.json) 있으면 건너뜀. 표: exe/k4why_tab.py -> $1/k4why_table.md. 끝 표식 $1/k4why.END.
O=$1; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
base() { bash -c "source exe/fem_cm8_scn.sh >/dev/null 2>&1; cm_scene L000 --v=b --ga hht --rho 0.8 --K $1 --newton 2 --cg 4 --tolN 1e-2 --tolP 1e-1 --det >/dev/null 2>&1; echo \"\$SIM \$DA \$XA\""; }
run() {    # run 이름 카드 K '뺄 인자' '더할 인자'
  local n=$1 g=$2 k=$3 drop=$4 add=$5 c
  [ -f $O/$n/summary.json ] && return
  c=" $(base $k) "
  for x in $drop; do c=${c// $x / }; done
  mkdir -p $O/$n; echo "$PY -u $c --tag L000 --out $O/$n $add" > $O/$n/cmd.txt
  echo "[$(date '+%T')] $n 시작" >> $O/k4why.log
  CUDA_VISIBLE_DEVICES=$g timeout 5400 $PY -u $c --tag L000 --out $O/$n $add < /dev/null > $O/$n/log.txt 2>&1
  echo "[$(date '+%T')] $n rc=$? $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')" >> $O/k4why.log
}
DW="--wdiag 40 --wdiag_end 68 --gfin --diag --pdiag --fhash --dump --dump_part"
DO="--fhash"
lane() { local g=$1; shift; while [ $# -gt 0 ]; do run $1 $g $2 "$3" "$4"; shift 4; done; }
# 두 줄 (같은 카드, 결정적 판이라 동시 실행이 결과를 안 바꿈)
lane 0 k4o 4 "" "$DO"   k4w 4 "" "$DW"   hso 4 "--hspec --ncomp" "$DO"   hsw 4 "--hspec --ncomp" "$DW" &
sleep 20
lane 0 k2o 2 "" "$DO"   k2w 2 "" "$DW"   mvo 4 "--mv2 --hspec --ncomp" "$DO"   mvw 4 "--mv2 --hspec --ncomp" "$DW" &
wait
$PY exe/k4why_tab.py $O 40 > $O/k4why_table.md 2> $O/k4why_table.err
echo "[$(date '+%T')] K4WHY_END" >> $O/k4why.log
echo "끝 $(date '+%F %T')" > $O/k4why.END
