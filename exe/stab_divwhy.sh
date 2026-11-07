#!/bin/bash
# 절 S.10: lego L000 에서 발산한 판 전부 -- '가벼운 노드 -> 칸 뒤집힘 -> 큰 복원력 -> 발산' 검증 (인스턴스 안, 커밋 디렉토리에서; 출력 $1, 기존 기록 $1/ref/<판> 은 vast_divwhy.sh 가 R2 에서 받아 둠).
# 판 (기존 기록 R2 lbours/...): k4 = stab_k4why/k4o (B) K 4 · fa·fu1·fu2 = fem_cm/astick/{a,u1,u2}_L000 (A)·(i)·(ii) FCR · sa·su1·su2 = fem_cm/astick_snh/{sa,su1,su2}_L000 SNH
#   · vs·vw = fem_cm/vres/{vs,vw}_L000 −z 저항 강·약 · nc = fem_cm/nclamp/nc_L000 노드 법선 클램프.
# 명령 = 기존 기록의 run_info.txt (k4 는 cmd.txt) 의 exe/fem_an_rt_sim.py 인자 그대로 + --tag L000 --out.
#   <판>o = + --fhash (진단 끔; 기존 기록과 프레임별 오차·노드 상자 같아야 재현), <판>w = + --wdiag 1 (모든 프레임 원자료) --pdiag --fhash (해시가 <판>o 와 같아야 결과 불변).
# 판마다 카드 하나를 혼자 씀 (메모리 부족으로 멈추는 판이 많아 동시 실행 금지). 카드 두 장에 줄 둘.
O=$1; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
cmd_of() { local r=$O/ref/$1 c
  if [ -f $r/run_info.txt ]; then c=$(head -1 $r/run_info.txt | sed 's/^.*exe\/fem_an_rt_sim.py/exe\/fem_an_rt_sim.py/')
  else c=$(head -1 $r/cmd.txt | sed 's/^.*exe\/fem_an_rt_sim.py/exe\/fem_an_rt_sim.py/; s/ --tag [^ ]*//; s/ --out [^ ]*//; s/ --fhash//'); fi
  echo "$c"; }
run() {    # run 판 카드 o|w
  local r=$1 g=$2 m=$3 n=$1$3 c add
  [ -f $O/$n/summary.json ] && return
  c=$(cmd_of $r); [ -n "$c" ] || { echo "[$(date '+%T')] $n 명령 없음" >> $O/divwhy.log; return; }
  if [ $m = o ]; then add="--fhash"; else add="--wdiag 1 --wdiag_end 1000 --pdiag --fhash"; fi
  rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --tag L000 --out $O/$n $add" > $O/$n/cmd.txt
  echo "[$(date '+%T')] $n 카드 $g 시작" >> $O/divwhy.log
  CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c --tag L000 --out $O/$n $add < /dev/null > $O/$n/log.txt 2>&1
  echo "[$(date '+%T')] $n rc=$? $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')" >> $O/divwhy.log
}
lane() { local g=$1; shift; for r in "$@"; do run $r $g o; run $r $g w; done; }
lane 0 ${DW_L0:-k4 fa fu1 sa su1} &
sleep 5
lane 1 ${DW_L1:-nc fu2 su2 vs vw} &
wait
$PY exe/divwhy_tab.py $O > $O/divwhy_table.md 2> $O/divwhy_table.err
echo "[$(date '+%T')] DIVWHY_END" >> $O/divwhy.log
echo "끝 $(date '+%F %T')" > $O/divwhy.END
