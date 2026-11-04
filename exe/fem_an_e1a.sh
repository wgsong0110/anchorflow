#!/bin/bash
# 절 AN.22 과제 e1a: HHT 장벽 판 1 뉴턴 발산 원인 진단 (--diag --gfin, 기존 코드 95fd7ed) + 2x5 커널 프로파일 (카드 0 단독)
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
C=95fd7ed; D=/dev/shm/dkta/impwt/$C
[ -f $D/.done ] || { mkdir -p $D && rclone copyto r2:storage/result/anchorflow/lbours/fem_an/code/imp_$C.tar.gz $D.tar.gz -q && tar -xzf $D.tar.gz -C $D && echo $C > $D/.done; }
cd $D || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_e1 XDG_CACHE_HOME=/dev/shm/dkta/cache; mkdir -p $CUPY_CACHE_DIR /dev/shm/dkta/an_lists
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mg_lreuse --mv2"
T="--tolN 1e-2 --tolP 1e-1"; T0="--tolN 0 --tolP 0"; TC="--tolN 1e-6 --tolP 1e-3"
H8="--ga hht --rho 0.8 --fl_bar --fl_bst 1 --gq_oct 0.1"
L=/dev/shm/dkta/an_lists/e1a.txt; : > $L
for t in L000 drop1 0016; do echo "g17 $t $B $H8 $T --newton 1 --cg 7 --diag --gfin" >> $L; done
for t in L000 drop1; do echo "g210 $t $B $H8 $T --newton 2 --cg 10 --diag --gfin" >> $L; done
echo "g840 drop1 $B $H8 $TC --newton 8 --cg 40 --diag --gfin" >> $L
for t in L000 drop1; do echo "gbe15 $t $B --integ be --fl_bar --fl_bst 1 --gq_oct 0.1 $T --newton 1 --cg 5 --diag --gfin" >> $L; done
( AN_CARDS="1 2 3 4 5 6" AN_PAR=6 AN_GAP=5 bash exe/fem_an_pool.sh e1a $L > /dev/shm/dkta/e1a_pool.log 2>&1; rclone copyto /dev/shm/dkta/e1a_pool.log r2:storage/result/anchorflow/lbours/fem_an/e1a/pool.log -q ) &
# 프로파일 (카드 0 단독)
SD=/dev/shm/dkta/anerr_data; RT=/dev/shm/dkta/lbours/fem_an/e1p; RRT=r2:storage/result/anchorflow/lbours/fem_an/e1p; mkdir -p $RT
U0=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
others0() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$U0"; }
exec 9>/home/dkta/work/gpu0.lock; flock 9
LG="--data $SD/L000.pt --floor vel --tag L000"
for vv in "p25:--newton 2 --cg 5 --prof 4" "k25:--newton 2 --cg 5 --kbench 6" "p17:--newton 1 --cg 7 --prof 4"; do v=${vv%%:*}; A=${vv#*:}; O=$RT/$v; mkdir -p $O
  while [ "$(others0)" -gt 0 ]; do sleep 5; done
  echo "시작 때 카드 0 의 다른 프로세스 $(others0) 개 $(date +%T)" > $O/card.txt
  CUDA_VISIBLE_DEVICES=0 timeout 1200 $PY -u exe/fem_an_rt_sim.py --out $O $B $LG --frames 10 $T0 $H8 $A 9>&- < /dev/null > $O/log.txt 2>&1
  echo "끝 때 카드 0 의 다른 프로세스 $(others0) 개 $(date +%T)" >> $O/card.txt
  rclone copy $O $RRT/$v -q; done
flock -u 9; exec 9>&-
wait
echo "[e1a 끝] $(date '+%F %T')"
