#!/bin/bash
# 절 AN.22 과제 e1d: PCG 수렴 저하 원인 가르기 (L000 2x5 --diag, 비결정): 장벽 유무·띠 고착 유무·mgbar·거친 층 ω 재사용 유무
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
C=22b66e1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_e1d XDG_CACHE_HOME=/dev/shm/dkta/cache; mkdir -p $CUPY_CACHE_DIR
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mv2 --lsfast --diag --gfin --tolN 1e-2 --tolP 1e-1 --newton 2 --cg 5 --gq_oct 0.1"
L=/dev/shm/dkta/an_lists/e1d.txt; : > $L
echo "dnob L000 $B --mg_lreuse --ga hht --rho 0.8" >> $L
echo "dbar0 L000 $B --mg_lreuse --ga hht --rho 0.8 --fl_bar" >> $L
echo "dbst L000 $B --mg_lreuse --ga hht --rho 0.8 --fl_bar --fl_bst 1" >> $L
echo "dbstnl L000 $B --ga hht --rho 0.8 --fl_bar --fl_bst 1" >> $L
echo "dbe L000 $B --mg_lreuse --integ be --fl_bar --fl_bst 1" >> $L
echo "dbenob L000 $B --mg_lreuse --integ be" >> $L
AN_CARDS="1 2 3 4 5 6" AN_PAR=6 AN_GAP=3 bash exe/fem_an_pool.sh e1d $L
