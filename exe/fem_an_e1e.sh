#!/bin/bash
# 절 AN.22 과제 e1e (커밋 $1): 뉴턴 반복별 PCG 상한 --cg_list (첫 뉴턴은 관성 몫이라 빨리 수렴, 둘째에 PCG 를 더) -- 판정 풀 (결정적) + 단독 시간
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_e1e XDG_CACHE_HOME=/dev/shm/dkta/cache; mkdir -p $CUPY_CACHE_DIR /dev/shm/dkta/an_lists
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mg_lreuse --mv2 --lsfast"
T="--tolN 1e-2 --tolP 1e-1"; T0="--tolN 0 --tolP 0"
H8="--ga hht --rho 0.8 --fl_bar --fl_bst 1 --gq_oct 0.1"
L=/dev/shm/dkta/an_lists/e1e.txt; : > $L
for cl in 2,4 1,5 3,4 2,6 2,8 3,7; do v=c${cl/,/}
  for t in L000 0016 0019 drop1; do echo "$v $t $B --det $H8 $T --newton 2 --cg 9 --cg_list $cl" >> $L; done; done
for t in L000 0016 0019 drop1; do echo "c28m $t $B --det $H8 $T --newton 2 --cg 9 --cg_list 2,8 --mgbar" >> $L; done
( AN_CARDS="1 2 3 4 5 6" AN_PAR=6 AN_GAP=4 bash exe/fem_an_pool.sh e1e $L > /dev/shm/dkta/e1e_pool.log 2>&1 ) &
Q=/dev/shm/dkta/an_lists/e1et.txt; : > $Q
for cl in 2,4 1,5 2,6 1,6 2,5; do echo "lw_c${cl/,/} @LG --frames 10 $T0 $H8 --newton 2 --cg 9 --cg_list $cl" >> $Q; done
bash exe/fem_an_e1q.sh e1et $Q
wait
echo "[e1e 끝] $(date '+%F %T')"
