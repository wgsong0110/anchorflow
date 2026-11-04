#!/bin/bash
# 절 CM6 셋째 묶음 cm6c (커밋 $1 = 9cdfe25): 뉴턴 반복별 |g|² (--diag) -- 장벽 판·고착 판·예전 판의 걸음 안 수렴 비교, 결정적, 빈 카드 1-6
C=$1; D=/dev/shm/dkta/impwt/$C; RC=r2:storage/result/anchorflow/lbours/fem_cm/code
export PATH=$HOME/bin:$PATH
if [ ! -f $D/.done ]; then
  mkdir -p /dev/shm/dkta/impwt; rm -rf $D.tmp
  rclone copyto $RC/imp_$C.bundle $D.bundle -q && git clone -q $D.bundle $D.tmp && git -C $D.tmp checkout -q $C && mv $D.tmp $D && echo bundle
  echo $C > $D/.done
fi
cd $D || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
W=/dev/shm/dkta/cm6c; mkdir -p $W; MR=r2:storage/result/anchorflow/lbours/fem_cm/marks; RR=r2:storage/result/anchorflow/lbours/fem_cm/cm6c
G="--ga hht --rho 0.8 --K 2"; T="--tolN 1e-2 --tolP 1e-1"; B="$G --newton 2 --cg 5 $T --det --hgt"; Z="--det --hgt"
L=$W/list.txt; cat > $L <<EOT
dr wolf $B --diag
dstk wolf $B --diag --no=--fl_bar --no=--fl_bst
dcv2 wolf $G --newton 20 --cg 60 --tolN 0 --tolP 0 $Z --diag
dcv2s wolf $G --newton 20 --cg 60 --tolN 0 --tolP 0 $Z --diag --no=--fl_bar --no=--fl_bst
dold wolf --ga hht --rho 0.7 --ga_y0ut --K 3 --newton 2 --cg 5 $T $Z --diag --noba
EOT
rclone copy $L $RR -q
CM_SCN=exe/fem_cm6_scn.sh CM_CARDS="1 2 3 4 5 6" CM_PAR=6 CM_GAP=5 CM_TMO=1500 bash exe/fem_cm_pool.sh cm6c $L > $W/pool.log 2>&1
rclone copyto $W/pool.log $MR/cm6c_pool.log -q
R=/dev/shm/dkta/lbours/fem_cm/cm6c
for v in ; do [ -f $R/${v}_wolf/pred.pt ] && rclone copy $R/${v}_wolf/pred.pt $RR/${v}_wolf -q; done
$PY exe/fem_cm6_sum.py $R > $W/cm6c_table.md 2>&1
rclone copyto $W/cm6c_table.md $MR/cm6c_table.md -q
date '+%F %T' > $W/cm6c.END; rclone copyto $W/cm6c.END $MR/cm6c.END -q
echo "[cm6c 끝] $(date '+%F %T')"
