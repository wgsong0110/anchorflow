#!/bin/bash
# 절 CM6 둘째 묶음 cm6b (커밋 $1): 장벽 판 wolf 가 안 무너지는 것이 덜 수렴 (허용치·반복 상한·장벽 헤시안 꼴) 인지 장벽 정의 (d̂) 인지 -- 인자 하나씩, 결정적, 빈 카드 1-6
C=$1; D=/dev/shm/dkta/impwt/$C; RC=r2:storage/result/anchorflow/lbours/fem_cm/code
export PATH=$HOME/bin:$PATH
if [ ! -f $D/.done ]; then
  mkdir -p /dev/shm/dkta/impwt; rm -rf $D.tmp
  rclone copyto $RC/imp_$C.bundle $D.bundle -q && git clone -q $D.bundle $D.tmp && git -C $D.tmp checkout -q $C && mv $D.tmp $D && echo bundle
  echo $C > $D/.done
fi
cd $D || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
W=/dev/shm/dkta/cm6b; mkdir -p $W; MR=r2:storage/result/anchorflow/lbours/fem_cm/marks; RR=r2:storage/result/anchorflow/lbours/fem_cm/cm6b
G="--ga hht --rho 0.8 --K 2"; T="--tolN 1e-2 --tolP 1e-1"; B="$G --newton 2 --cg 5 $T --det --hgt"; Z="--det --hgt"
L=$W/list.txt; cat > $L <<EOT
p0 wolf $G --newton 2 --cg 5 --tolN 1e-2 --tolP 0 $Z
p0n0 wolf $G --newton 2 --cg 5 --tolN 0 --tolP 0 $Z
p0c20 wolf $G --newton 2 --cg 20 --tolN 1e-2 --tolP 0 $Z
cv2 wolf $G --newton 20 --cg 60 --tolN 0 --tolP 0 $Z --dump
cv2n wolf $G --newton 20 --cg 60 --tolN 0 --tolP 0 $Z --fl_nost
cv2s wolf $G --newton 20 --cg 60 --tolN 0 --tolP 0 $Z --no=--fl_bar --no=--fl_bst
bf wolf $B --bar_full
bfcv2 wolf $G --newton 20 --cg 60 --tolN 0 --tolP 0 $Z --bar_full
mg0 wolf $B --mgbar 0
mg1 wolf $B --mgbar 1
dh4 wolf $B --fl_dhm 4 --dump
dh2 wolf $B --fl_dhm 2
k4 wolf --ga hht --rho 0.8 --K 4 --newton 2 --cg 5 $T $Z
s0p0 wolf $G --newton 2 --cg 5 --tolN 1e-2 --tolP 0 $Z --no=--fl_bar --no=--fl_bst
EOT
rclone copy $L $RR -q
CM_SCN=exe/fem_cm6_scn.sh CM_CARDS="1 2 3 4 5 6" CM_PAR=6 CM_GAP=5 CM_TMO=1500 bash exe/fem_cm_pool.sh cm6b $L > $W/pool.log 2>&1
rclone copyto $W/pool.log $MR/cm6b_pool.log -q
R=/dev/shm/dkta/lbours/fem_cm/cm6b
for v in cv2 dh4; do [ -f $R/${v}_wolf/pred.pt ] && rclone copy $R/${v}_wolf/pred.pt $RR/${v}_wolf -q; done
$PY exe/fem_cm6_sum.py $R > $W/cm6b_table.md 2>&1
rclone copyto $W/cm6b_table.md $MR/cm6b_table.md -q
date '+%F %T' > $W/cm6b.END; rclone copyto $W/cm6b.END $MR/cm6b.END -q
echo "[cm6b 끝] $(date '+%F %T')"
