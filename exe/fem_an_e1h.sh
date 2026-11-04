#!/bin/bash
# 절 AN.22 과제 e1h: 단독 시간 다시 (e1f·e1g 시간 목록에 --lsfast 가 빠졌음) -- 커밋 3e96de6, --lsfast 포함
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
C=3e96de6; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
T="--tolN 1e-2 --tolP 1e-1"; T0="--tolN 0 --tolP 0"
H8="--ga hht --rho 0.8 --fl_bar --fl_bst 1 --gq_oct 0.1 --mgbar 2 --lsfast"
Q=/dev/shm/dkta/an_lists/e1ht.txt; : > $Q
for v in "h23:--newton 2 --cg 3" "h24:--newton 2 --cg 4" "h25:--newton 2 --cg 5"; do echo "lw_${v%%:*} @LG --frames 10 $T0 $H8 ${v#*:}" >> $Q; done
echo "dw_h23 @DR --frames 10 $T0 $H8 --newton 2 --cg 3" >> $Q
echo "lr_h23 @LG --frames 80 $T $H8 --newton 2 --cg 3" >> $Q
echo "dr_h23 @DR --frames 80 $T $H8 --newton 2 --cg 3" >> $Q
echo "lr_h24 @LG --frames 80 $T $H8 --newton 2 --cg 4" >> $Q
echo "fw_h220 @FI --frames 20 $T0 --ga hht --rho 0.8 --gq_oct 0.1 --lsfast --newton 2 --cg 20" >> $Q
echo "fr_h220 @FI --frames 125 $T --ga hht --rho 0.8 --gq_oct 0.1 --lsfast --newton 2 --cg 20" >> $Q
echo "pf_h23 @LG --frames 6 --prof 4 $T0 $H8 --newton 2 --cg 3" >> $Q
bash exe/fem_an_e1q.sh e1ht $Q
