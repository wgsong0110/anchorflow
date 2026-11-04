#!/bin/bash
# 절 AN.22 과제 e1j (커밋 $1): e1i 의 단독 시간 목록에 --lsfast --fk 가 빠져 (fem_an_e1q.sh 의 공통 앞 인자에 없음) 다시 잼 -> e1t2; e1i 끝난 뒤 표 marks/e1_table.md 를 e1t2 로 다시 씀
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
M=/dev/shm/dkta/lbours/fem_an/marks; MR=r2:storage/result/anchorflow/lbours/fem_an/marks
LOG=$M/e1.log; say() { echo "$*" | tee -a $LOG; rclone copy $LOG $MR -q; }
T="--tolN 1e-2 --tolP 1e-1"; T0="--tolN 0 --tolP 0"
H8="--ga hht --rho 0.8 --fl_bar --fl_bst 1 --gq_oct 0.1 --mgbar 2 --lsfast --fk"
FIA="--ga hht --rho 0.8 --gq_oct 0.1 --mgbar 2 --lsfast --fk --newton 2 --cg 20"
Q=/dev/shm/dkta/an_lists/e1t2.txt; : > $Q
for r in 1 2; do echo "lw_k23_r$r @LG --frames 10 $T0 $H8 --newton 2 --cg 3" >> $Q; done
echo "dw_k23 @DR --frames 10 $T0 $H8 --newton 2 --cg 3" >> $Q
echo "fw_k220 @FI --frames 20 $T0 $FIA" >> $Q
echo "fr_k220 @FI --frames 125 $T $FIA" >> $Q
echo "dr_k23 @DR --frames 80 $T $H8 --newton 2 --cg 3" >> $Q
echo "lr_k23_L000 @LG --frames 80 $T $H8 --newton 2 --cg 3" >> $Q
LD=/tmp/dkta/af_data; [ -f $LD/0019.pt ] || LD=/dev/shm/dkta/anerr_data
for t in $(seq -f %04g 0 19); do [ -s $LD/$t.pt ] || cp /home/dkta/work/lbpt/test/$t.pt $LD/$t.pt; echo "lr_k23_$t --data $LD/$t.pt --floor vel --tag $t --frames 80 $T $H8 --newton 2 --cg 3" >> $Q; done
echo "pf_k23 @LG --frames 6 --prof 4 $T0 $H8 --newton 2 --cg 3" >> $Q
bash exe/fem_an_e1q.sh e1t2 $Q > $M/e1t2_q.log 2>&1
cp /dev/shm/dkta/lbours/fem_an/e1t2/run.out $M/e1_time.md; rclone copy $M/e1_time.md $MR -q
say "[시간 다시 끝 e1t2] $(date '+%F %T')"
while [ ! -f $M/e1.END ]; do sleep 30; done
{ echo "# e1 ($C) -- 절 AN.22 최종 공통 설정 판정·시간"; echo
  echo "## 판정 (결정적; lego·낙하 상한 2x3, ficus 2x20)"; $PY exe/fem_an_tol_table.py /dev/shm/dkta/lbours/fem_an/e1
  for d in /dev/shm/dkta/lbours/fem_an/e1/*_drop1; do v=$(basename $d); echo "- ${v%_drop1} 낙하: $(tail -1 $d/drop_ana.txt 2>/dev/null | cut -c1-220)"; done
  for d in /dev/shm/dkta/lbours/fem_an/e1/*_ficus; do echo "- ficus: $(tail -2 $d/period.txt 2>/dev/null | tr '\n' ' ' | cut -c1-300)"; done
  echo; echo "## 단독 시간 (카드 0, 비결정, e1t2; lw = 최악 허용치 0 10 프레임, lr·dr·fr = 실제 공통 허용치 전체 프레임, 첫 2 프레임 뺀 평균·최대)"; cat $M/e1_time.md
  echo; echo "(e1t 는 --lsfast --fk 가 빠진 시간이라 무효)"; } > $M/e1_table.md 2>&1
rclone copy $M $MR -q --include "e1*"
say "[e1 표 다시 씀] $(date '+%F %T')"
