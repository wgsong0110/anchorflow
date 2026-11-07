#!/bin/bash
# 절 S.11: lego L000 에서 발산했던 판에 --gq_mass (절 CM9: 질량·운동량을 입자 -> 팔분 가우스 점 -> 노드) 만 켜 봄 + (B) K 2 (채택 판) 회귀 (잡 안, 커밋 디렉토리에서; 출력 $1).
# 판 (원래 명령 = 기존 기록 그대로, 모두 결정적 --det; 데이터 /dev/shm/dkta/anerr_data/L000.pt):
#   (B) 장벽 바닥 K 4·6·8·12·16 = 최소 안정 서브스텝 실험 명령 (exe/stab_k4diag.sh base = cm_scene L000 --v=b, 뉴턴 2 x PCG 4), K 만 바꿈 -- 기록 R2 lbours/stab_k4diag/K<k>, stab_k4why/k4o
#   (A) 원래 고착 fa · (A)(i) fu1 · (A)(ii) fu2 (FCR) · −z 저항 강 vs (k 9)·약 vw (k 0.9) · 노드 −z 클램프 nc -- 기록 R2 lbours/divwhy/<판>o/cmd.txt (= fem_cm/astick·vres·nclamp)
#   (B) K 2 (채택 판) -- 기록 stab_k4why/k2o (끝 해시 c43a42c7d0, 3.62 %)
# 실행: <판>_off_w = 원래 명령 + 진단 (--wdiag 1 --wdiag_end 1000 --pdiag --fhash; 결과 불변), <판>_on_w = + --gq_mass --gq_chk + 같은 진단, <판>_on_o = + --gq_mass --fhash 만 (켬 판 진단 불변 확인).
# 카드 7 장, 카드 하나에 판 하나 (메모리 부족 판이 많아 한 카드 동시 실행 금지), 공용 대기열. 끝나고 K 2 단독 시간 (비결정 = --det 뺌, 카드 0, 노드에 다른 판 없음; 끔·켬 번갈아 두 번).
# 판 결과 (summary.json) 있으면 건너뜀. 표: exe/gqm_tab.py -> $1/gqm_table.md, 끝 표식 $1/gqm.END.
O=$1; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
SD=/dev/shm/dkta/anerr_data; mkdir -p $SD
[ -s $SD/L000.pt ] || { cp /home/dkta/work/one/L000/L000.pt $SD/L000.pt.$$ && mv $SD/L000.pt.$$ $SD/L000.pt; }
D="--data $SD/L000.pt --frames 80"
BB="--floor vel --fl_bar --fl_bst 1 --y0eb --cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2 --ga_y0ut --hspec --ncomp --ga hht --rho 0.8"
BA="--floor vel --cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2 --ga_y0ut --hspec --ncomp --ga hht --rho 0.8"
TL="--tolN 1e-2 --tolP 1e-1 --det"
cmd_of() { case $1 in
  K2|K4|K6|K8|K12|K16) echo "$D $BB --K ${1#K} --newton 2 --cg 4 $TL";;
  fa)  echo "$D $BA --K 2 --newton 2 --cg 6 $TL --pen --hgt --pdiag --dump --dump_F --dump_part";;
  fu1) echo "$D $BA --K 2 --newton 2 --cg 6 $TL --pen --hgt --pdiag --dump --dump_F --dump_part --fl_up 1";;
  fu2) echo "$D $BA --K 2 --newton 2 --cg 6 $TL --pen --hgt --pdiag --dump --dump_F --dump_part --fl_up 2";;
  vs)  echo "$D $BA --K 2 --newton 2 --cg 4 $TL --hgt --pen --dump --dump_F --dump_part --fl_vres 9";;
  vw)  echo "$D $BA --K 2 --newton 2 --cg 4 $TL --hgt --pen --dump --dump_F --dump_part --fl_vres 0.9";;
  nc)  echo "$D $BA --K 2 --newton 2 --cg 4 $TL --hgt --pen --dump --dump_F --dump_part --fl_nclamp";;
esac; }
WD="--wdiag 1 --wdiag_end 1000 --pdiag --fhash"
add_of() { case $1 in off_w) echo "$WD";; on_w) echo "--gq_mass --gq_chk $WD";; on_o) echo "--gq_mass --fhash";; esac; }
run() {    # run 판 모드 카드
  local r=$1 m=$2 g=$3 n=$1_$2 c a
  [ -f $O/$n/summary.json ] && return
  c=$(cmd_of $r); a=$(add_of $m)
  rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u exe/fem_an_rt_sim.py $c --tag L000 --out $O/$n $a" > $O/$n/cmd.txt
  echo "[$(date '+%F %T')] $n 카드 $g 시작" >> $O/gqm.log
  CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u exe/fem_an_rt_sim.py $c --tag L000 --out $O/$n $a < /dev/null > $O/$n/log.txt 2>&1
  echo "[$(date '+%F %T')] $n rc=$? $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')" >> $O/gqm.log
}
# ---- 연기 (4 프레임, 카드 0~4): 켬 판 컴파일·진단 불변 (o 대 w 프레임별 해시)
SM=$O/smoke; mkdir -p $SM
smk() { local r=$1 m=$2 g=$3 c a S=$SM/$1_$2; mkdir -p $S
  c=$(cmd_of $r | sed 's/--frames 80/--frames 4/'); a=$(add_of $m)
  echo "$PY -u exe/fem_an_rt_sim.py $c --tag L000 --out $S $a" > $S/cmd.txt
  CUDA_VISIBLE_DEVICES=$g timeout 1800 $PY -u exe/fem_an_rt_sim.py $c --tag L000 --out $S $a < /dev/null > $S/log.txt 2>&1; echo "rc=$?" > $S/rc.txt; }
if [ ! -f $SM/ok ]; then
  i=0; for r in K4 fa nc vs K2; do ( smk $r on_o $i; smk $r on_w $i ) & i=$((i + 1)); sleep 3; done; wait
  OK=1; MSG=""
  for r in K4 fa nc vs K2; do
    HO=$(grep -o '"xh": "[0-9a-f]*"' $SM/${r}_on_o/fem.jsonl 2>/dev/null | tr '\n' ' '); HW=$(grep -o '"xh": "[0-9a-f]*"' $SM/${r}_on_w/fem.jsonl 2>/dev/null | tr '\n' ' ')
    GC=$(grep -o '"gqchk": \[[^]]*\]' $SM/${r}_on_w/fem.jsonl 2>/dev/null | tail -1)
    if [ -n "$HO" ] && [ "$HO" = "$HW" ] && [ $(echo $HO | wc -w) -eq 4 ]; then MSG="$MSG | $r 해시 같음 $GC"; else OK=0; MSG="$MSG | $r 다름/없음 o[$HO] w[$HW] $(tail -3 $SM/${r}_on_o/log.txt | tr '\n' ' ' | cut -c1-300)"; fi
  done
  echo "[$(date '+%F %T')] [연기] OK=$OK $MSG" >> $O/gqm.log
  [ $OK = 1 ] || { echo "연기 실패 $(date '+%F %T')" > $O/gqm.END; exit 1; }
  touch $SM/ok
fi
# ---- 본 묶음: 공용 대기열 (긴 판 먼저)
Q=$O/queue.txt; [ -f $Q ] || {
  for r in K2 nc K4 K6 K16 K8 fu2 vs vw fu1 fa K12; do echo "$r on_w"; echo "$r off_w"; done
  for r in K2 nc K4 K6 K16 K8 fu2 vs vw fu1 fa K12; do echo "$r on_o"; done; } > $Q
echo 0 > $O/qpos
lane() { local g=$1 i ln
  while :; do
    i=$(flock $O/qpos.lock bash -c "i=\$(cat $O/qpos); echo \$((i + 1)) > $O/qpos; echo \$i")
    ln=$(sed -n "$((i + 1))p" $Q); [ -n "$ln" ] || break
    run $ln $g
  done; }
for g in 0 1 2 3 4 5 6; do lane $g & sleep 2; done
wait
echo "[$(date '+%F %T')] 판정 판 끝" >> $O/gqm.log
# ---- K 2 단독 시간 (비결정 판 = 판정 명령에서 --det 만 뺌; 카드 0, 노드에 다른 판 없음)
zrun() { local n=$1 a=$2 c
  [ -f $O/$n/summary.json ] && return
  c=$(cmd_of K2 | sed 's/ --det//'); mkdir -p $O/$n
  echo "$PY -u exe/fem_an_rt_sim.py $c --tag L000 --out $O/$n $a" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > $O/$n/gpu_before.txt 2>&1
  ( while sleep 10; do echo "$(date '+%T') $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" >> $O/$n/gpu_procs.txt; done ) & local wp=$!
  CUDA_VISIBLE_DEVICES=0 timeout 3600 $PY -u exe/fem_an_rt_sim.py $c --tag L000 --out $O/$n $a < /dev/null > $O/$n/log.txt 2>&1
  kill $wp 2>/dev/null
  echo "[$(date '+%F %T')] $n $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200)" >> $O/gqm.log; }
zrun z_off_a ""; zrun z_on_a "--gq_mass"; zrun z_off_b ""; zrun z_on_b "--gq_mass"
$PY exe/gqm_tab.py $O > $O/gqm_table.md 2> $O/gqm_table.err
echo "[$(date '+%F %T')] GQM_END" >> $O/gqm.log
echo "끝 $(date '+%F %T')" > $O/gqm.END
