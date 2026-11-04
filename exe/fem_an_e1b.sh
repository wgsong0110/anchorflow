#!/bin/bash
# 절 AN.22 과제 e1b (커밋 $1): (1) 판정 풀 (카드 1-6, 결정적): 시작점 탐색 --y0ls 의 HHT ρ0.8 장벽 띠 고착 1x5·1x7·2x5, 선탐색 조기 판정 --lsfast 결과 같음 확인 (해시), 진단
#                        (2) 단독 시간 (카드 0): --lsfast·--y0ls 최악 (허용치 0) lego L000 10 프레임
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
C=$1; D=/dev/shm/dkta/impwt/$C
[ -f $D/.done ] || { mkdir -p $D && rclone copyto r2:storage/result/anchorflow/lbours/fem_an/code/imp_$C.tar.gz $D.tar.gz -q && tar -xzf $D.tar.gz -C $D && echo $C > $D/.done; }
cd $D || exit 1
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_e1b XDG_CACHE_HOME=/dev/shm/dkta/cache; mkdir -p $CUPY_CACHE_DIR /dev/shm/dkta/an_lists
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mg_lreuse --mv2"
T="--tolN 1e-2 --tolP 1e-1"; T0="--tolN 0 --tolP 0"
H8="--ga hht --rho 0.8 --fl_bar --fl_bst 1 --gq_oct 0.1"
L=/dev/shm/dkta/an_lists/e1b.txt; : > $L
echo "eq17 L000 $B --det $H8 $T --newton 1 --cg 7" >> $L
echo "eq17f L000 $B --det $H8 $T --newton 1 --cg 7 --lsfast" >> $L
echo "eq210f L000 $B --det $H8 $T --newton 2 --cg 10 --lsfast" >> $L
for cap in 1x5 1x7 2x5; do n=${cap%x*}; c=${cap#*x}
  for t in L000 0016 0019 drop1; do echo "y$cap $t $B --det $H8 $T --newton $n --cg $c --lsfast --y0ls" >> $L; done; done
for t in L000 0016 0019 drop1; do echo "yf1x5 $t $B --det $H8 --bar_full $T --newton 1 --cg 5 --lsfast --y0ls" >> $L; done
for t in L000 0016 0019 drop1; do echo "ybe1x5 $t $B --det --integ be --fl_bar --fl_bst 1 --gq_oct 0.1 $T --newton 1 --cg 5 --lsfast --y0ls" >> $L; done
for t in L000 drop1; do echo "yd1x7 $t $B $H8 $T --newton 1 --cg 7 --lsfast --y0ls --diag --gfin" >> $L; done
( AN_CARDS="1 2 3 4 5 6" AN_PAR=6 AN_GAP=5 bash exe/fem_an_pool.sh e1b $L > /dev/shm/dkta/e1b_pool.log 2>&1; rclone copyto /dev/shm/dkta/e1b_pool.log r2:storage/result/anchorflow/lbours/fem_an/e1b/pool.log -q ) &
Q=/dev/shm/dkta/an_lists/e1bt.txt; : > $Q
for v in "r17:--newton 1 --cg 7" "f17:--newton 1 --cg 7 --lsfast" "f25:--newton 2 --cg 5 --lsfast" "f24:--newton 2 --cg 4 --lsfast" "f23:--newton 2 --cg 3 --lsfast" \
         "y15:--newton 1 --cg 5 --lsfast --y0ls" "y17:--newton 1 --cg 7 --lsfast --y0ls" "y25:--newton 2 --cg 5 --lsfast --y0ls" "r25:--newton 2 --cg 5"; do
  echo "lw_${v%%:*} @LG --frames 10 $T0 $H8 ${v#*:}" >> $Q; done
echo "kb_f25 @LG --frames 6 --kbench 6 $T0 $H8 --newton 2 --cg 5 --lsfast" >> $Q
echo "pf_y25 @LG --frames 6 --prof 4 $T0 $H8 --newton 2 --cg 5 --lsfast --y0ls" >> $Q
bash exe/fem_an_e1q.sh e1bt $Q
wait
echo "[e1b 끝] $(date '+%F %T')"
