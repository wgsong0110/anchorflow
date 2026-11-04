#!/bin/bash
# 절 AN.22 과제 e1g (커밋 $1): --mgbar 장벽 몫을 노드 연산·rt_ap 에 융합 (결과 같은 식, 커널 수 줄임) -- 판정 (결정적) 2x3·2x4 + 단독 시간 (최악 10 프레임·실제 80 프레임·프로파일)
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_e1g XDG_CACHE_HOME=/dev/shm/dkta/cache; mkdir -p $CUPY_CACHE_DIR /dev/shm/dkta/an_lists
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mg_lreuse --mv2 --lsfast"
T="--tolN 1e-2 --tolP 1e-1"; T0="--tolN 0 --tolP 0"
H8="--ga hht --rho 0.8 --fl_bar --fl_bst 1 --gq_oct 0.1 --mgbar 2"
L=/dev/shm/dkta/an_lists/e1g.txt; : > $L
for cap in 2x3 2x4; do n=${cap%x*}; c=${cap#*x}
  for t in L000 0016 0019 drop1; do echo "h$cap $t $B --det $H8 $T --newton $n --cg $c" >> $L; done; done
echo "fih8l ficus $B --det --ga hht --rho 0.8 --gq_oct 0.1 $T --newton 2 --cg 20" >> $L
( AN_CARDS="1 2 3 4 5 6" AN_PAR=6 AN_GAP=4 bash exe/fem_an_pool.sh e1g $L > /dev/shm/dkta/e1g_pool.log 2>&1 ) &
Q=/dev/shm/dkta/an_lists/e1gt.txt; : > $Q
for v in "h23:--newton 2 --cg 3" "h24:--newton 2 --cg 4" "h17:--newton 1 --cg 7"; do echo "lw_${v%%:*} @LG --frames 10 $T0 $H8 ${v#*:}" >> $Q; done
echo "lr_h23 @LG --frames 80 $T $H8 --newton 2 --cg 3" >> $Q
echo "dr_h23 @DR --frames 80 $T $H8 --newton 2 --cg 3" >> $Q
echo "fw_h220 @FI --frames 20 $T0 --ga hht --rho 0.8 --gq_oct 0.1 --newton 2 --cg 20" >> $Q
echo "fr_h220 @FI --frames 125 $T --ga hht --rho 0.8 --gq_oct 0.1 --newton 2 --cg 20" >> $Q
echo "pf_h23 @LG --frames 6 --prof 4 $T0 $H8 --newton 2 --cg 3" >> $Q
bash exe/fem_an_e1q.sh e1gt $Q
wait
echo "[e1g 끝] $(date '+%F %T')"
