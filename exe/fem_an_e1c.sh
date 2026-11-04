#!/bin/bash
# 절 AN.22 과제 e1c (커밋 $1): MG 전처리 고운 층에 장벽 헤시안 (--mgbar, 전처리만) -- 판정 풀 (카드 1-6, 결정적) HHT ρ0.8 띠 고착 상한 1x5·1x7·2x3·2x4·2x5 + 단독 시간 (카드 0)
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_e1c XDG_CACHE_HOME=/dev/shm/dkta/cache; mkdir -p $CUPY_CACHE_DIR /dev/shm/dkta/an_lists
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mg_lreuse --mv2"
T="--tolN 1e-2 --tolP 1e-1"; T0="--tolN 0 --tolP 0"
H8="--ga hht --rho 0.8 --fl_bar --fl_bst 1 --gq_oct 0.1"
L=/dev/shm/dkta/an_lists/e1c.txt; : > $L
for cap in 2x3 2x4 2x5 1x7 1x5; do n=${cap%x*}; c=${cap#*x}
  for t in L000 0016 0019 drop1; do echo "m$cap $t $B --det $H8 $T --newton $n --cg $c --lsfast --mgbar" >> $L; done; done
for t in L000 0016 0019 drop1; do echo "mf2x4 $t $B --det $H8 --bar_full $T --newton 2 --cg 4 --lsfast --mgbar" >> $L; done
for t in L000 0016 0019 drop1; do echo "mbe1x5 $t $B --det --integ be --fl_bar --fl_bst 1 --gq_oct 0.1 $T --newton 1 --cg 5 --lsfast --mgbar" >> $L; done
echo "md17 L000 $B $H8 $T --newton 1 --cg 7 --lsfast --mgbar --diag --gfin" >> $L
echo "md25 L000 $B $H8 $T --newton 2 --cg 5 --lsfast --mgbar --diag --gfin" >> $L
( AN_CARDS="1 2 3 4 5 6" AN_PAR=6 AN_GAP=5 bash exe/fem_an_pool.sh e1c $L > /dev/shm/dkta/e1c_pool.log 2>&1; rclone copyto /dev/shm/dkta/e1c_pool.log r2:storage/result/anchorflow/lbours/fem_an/e1c/pool.log -q ) &
Q=/dev/shm/dkta/an_lists/e1ct.txt; : > $Q
for v in "m23:--newton 2 --cg 3" "m24:--newton 2 --cg 4" "m25:--newton 2 --cg 5" "m17:--newton 1 --cg 7" "m26:--newton 2 --cg 6"; do
  echo "lw_${v%%:*} @LG --frames 10 $T0 $H8 --lsfast --mgbar ${v#*:}" >> $Q; done
echo "dw_m24 @DR --frames 10 $T0 $H8 --lsfast --mgbar --newton 2 --cg 4" >> $Q
bash exe/fem_an_e1q.sh e1ct $Q
wait
echo "[e1c 끝] $(date '+%F %T')"
