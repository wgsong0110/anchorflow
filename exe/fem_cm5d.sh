#!/bin/bash
# 절 CM5 진단 묶음 cm5d (코드 866d4e1 = cm4 와 같은 코드, 정의 변경 없음): cm4 발산 6 궤적 + L000 대조, 인자 하나씩 바꿈 (결정적)
C=866d4e1; D=/dev/shm/dkta/impwt/$C; RC=r2:storage/result/anchorflow/lbours/fem_cm/code
export PATH=$HOME/bin:$PATH
if [ ! -f $D/.done ]; then
  mkdir -p /dev/shm/dkta/impwt; rm -rf $D.tmp
  rclone copyto $RC/imp_$C.bundle $D.bundle -q && git clone -q $D.bundle $D.tmp && git -C $D.tmp checkout -q $C && mv $D.tmp $D && echo bundle
  echo $C > $D/.done
fi
cd $D || exit 1
W=/dev/shm/dkta/cm5d; mkdir -p $W; MR=r2:storage/result/anchorflow/lbours/fem_cm/marks
rclone copyto r2:storage/result/anchorflow/lbours/fem_cm/w2/aux/cm5d_scn.sh $W/cm5d_scn.sh -q
G="--ga hht --rho 0.8 --K 2 --newton 2"; T="--tolN 1e-2 --tolP 1e-1"
L=$W/list.txt; : > $L
for t in 0015 0017 0018 0001 0006 0014; do
  echo "r $t $G --cg 6 $T --det --diag" >> $L        # 재현 + 반복별 진단
  echo "c3 $t $G --cg 3 $T --det" >> $L               # cm2 상한
  echo "c12 $t $G --cg 12 $T --det" >> $L
  echo "c24 $t $G --cg 24 $T --det" >> $L
  echo "p0 $t $G --cg 6 --tolN 1e-2 --tolP 0 --det" >> $L   # PCG 허용치 0 (상한까지)
  echo "n4 $t --ga hht --rho 0.8 --K 2 --newton 4 --cg 6 $T --det" >> $L
  echo "u $t $G --cg 6 $T --det --U0 --diag" >> $L    # 시작점 U (진단 전용)
  echo "u3 $t $G --cg 3 $T --det --U0" >> $L
done
echo "r L000 $G --cg 6 $T --det --diag" >> $L
rclone copy $L r2:storage/result/anchorflow/lbours/fem_cm/cm5d -q
CM_SCN=$W/cm5d_scn.sh CM_CARDS="1 2 3 4 5 6" CM_PAR=6 CM_GAP=5 CM_TMO=900 bash exe/fem_cm_pool.sh cm5d $L > $W/pool.log 2>&1
rclone copyto $W/pool.log $MR/cm5d_pool.log -q
echo "[cm5d 끝] $(date '+%F %T')"
