#!/bin/bash
# 절 AN.22 과제 e1f (커밋 $1): 거친 층 장벽 --mgbar 2 (전처리만) -- PCG 수렴 진단 (L000 2x5 --diag) + 판정 풀 (결정적) 2x3·2x4·2x5 + 단독 시간
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_e1f XDG_CACHE_HOME=/dev/shm/dkta/cache; mkdir -p $CUPY_CACHE_DIR /dev/shm/dkta/an_lists
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mg_lreuse --mv2 --lsfast"
T="--tolN 1e-2 --tolP 1e-1"; T0="--tolN 0 --tolP 0"
H8="--ga hht --rho 0.8 --fl_bar --fl_bst 1 --gq_oct 0.1"
L=/dev/shm/dkta/an_lists/e1f.txt; : > $L
echo "fd25m2 L000 $B $H8 $T --newton 2 --cg 5 --mgbar 2 --diag --gfin" >> $L
echo "fd25m0 L000 $B $H8 $T --newton 2 --cg 5 --diag --gfin" >> $L
echo "fd210m2 L000 $B $H8 $T --newton 2 --cg 10 --mgbar 2 --diag --gfin" >> $L
for cap in 2x5 2x4 2x3 1x7; do n=${cap%x*}; c=${cap#*x}
  for t in L000 0016 0019 drop1; do echo "g$cap $t $B --det $H8 $T --newton $n --cg $c --mgbar 2" >> $L; done; done
( AN_CARDS="1 2 3 4 5 6" AN_PAR=6 AN_GAP=4 bash exe/fem_an_pool.sh e1f $L > /dev/shm/dkta/e1f_pool.log 2>&1 ) &
Q=/dev/shm/dkta/an_lists/e1ft.txt; : > $Q
for v in "g23:--newton 2 --cg 3" "g24:--newton 2 --cg 4" "g25:--newton 2 --cg 5" "g17:--newton 1 --cg 7"; do echo "lw_${v%%:*} @LG --frames 10 $T0 $H8 --mgbar 2 ${v#*:}" >> $Q; done
bash exe/fem_an_e1q.sh e1ft $Q
wait
echo "[e1f 끝] $(date '+%F %T')"
