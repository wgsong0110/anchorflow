#!/bin/bash
# 절 AN.19 묶음 b1 (잡 안, 비측정, 결정적): 장벽 바닥 (--fl_bar) x 후퇴 K 2 · HHT ρ 0.7·0.8·0.9 K 2, 공통 허용치 εN 1e-2 · εP 1e-1, 상한 lego·낙하·블록 1 x 5, ficus 2 x 10.
# 1 단계: 낙하·lego 0016·0019·0003·L000 (0016 덤프) -> 표 marks/b1a_table.md, 0016 접촉 분석 b1a_diag.txt, 표식 b1a.END
# 2 단계: 후퇴·HHT 0.8·0.9 의 나머지 lego 21·ficus·블록 신축/전단 -> 표 marks/b1_table.md, 표식 b1.END
cd "$(dirname "$0")/.." || exit 1
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1
PY=/home/dkta/.conda/envs/af/bin/python
M=/dev/shm/dkta/lbours/fem_an/marks; mkdir -p $M /dev/shm/dkta/an_lists
R=/dev/shm/dkta/lbours/fem_an/b1
TJ=/tmp/dkta/af_data/0016.pt; [ -s $TJ ] || TJ=/dev/shm/dkta/anerr_data/0016.pt
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mg_lreuse --det --fl_bar --tolN 1e-2 --tolP 1e-1"
VS=("bbe:--integ be" "bh8:--ga hht --rho 0.8" "bh9:--ga hht --rho 0.9" "bh7:--ga hht --rho 0.7")
L=/dev/shm/dkta/an_lists/b1a.txt; : > $L
for vv in "${VS[@]}"; do v=${vv%%:*}; I=${vv#*:}
  for t in drop1 0016 0019 0003 L000; do D=""; [ $t = 0016 ] && D="--dump"; echo "$v $t $B $I --newton 1 --cg 5 $D" >> $L; done; done
echo "[b1 시작] $(cat .done) $(date '+%F %T')" > $M/b1.log
AN_CARDS="1 2 3 4 5 6" AN_PAR=6 AN_GAP=5 bash exe/fem_an_pool.sh b1 $L > $M/b1a_pool.log 2>&1
{ $PY exe/fem_an_tol_table.py $R; for vv in "${VS[@]}"; do v=${vv%%:*}; echo "## $v 낙하 $(tail -1 $R/${v}_drop1/drop_ana.txt 2>/dev/null | cut -c1-230)"; grep -a "rror\|발산" $R/${v}_*/log.txt | head -3; done; } > $M/b1a_table.md 2>&1
A=""; for vv in "${VS[@]}"; do v=${vv%%:*}; [ -s $R/${v}_0016/pred.pt ] && A="$A $v:$R/${v}_0016/pred.pt"; done
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 $PY -u exe/fem_an_contact_ana.py $TJ $A > $M/b1a_diag.txt 2>&1
echo "[b1a 끝] $(date '+%F %T')" > $M/b1a.END; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
L=/dev/shm/dkta/an_lists/b1b.txt; : > $L
for vv in "${VS[@]:0:3}"; do v=${vv%%:*}; I=${vv#*:}
  echo "$v ficus $B $I --newton 2 --cg 10" >> $L
  for k in stretch shear; do echo "$v $k $B $I --newton 1 --cg 5" >> $L; done
  for t in $(seq -f %04g 0 19); do case $t in 0016|0019|0003) continue;; esac; echo "$v $t $B $I --newton 1 --cg 5" >> $L; done; done
AN_CARDS="1 2 3 4 5 6" AN_PAR=6 AN_GAP=5 bash exe/fem_an_pool.sh b1 $L > $M/b1b_pool.log 2>&1
{ $PY exe/fem_an_tol_table.py $R; for vv in "${VS[@]}"; do v=${vv%%:*}; echo "## $v 낙하 $(tail -1 $R/${v}_drop1/drop_ana.txt 2>/dev/null | cut -c1-230)"; done; } > $M/b1_table.md 2>&1
echo "[b1 끝] $(date '+%F %T')" > $M/b1.END; rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q
