#!/bin/bash
# 절 CM6 진단 묶음 cm6 (커밋 $1, 잡 130567 일꾼 w2): wolf 모래가 무너지지 않는 원인 -- 절 CM4 채택 공통 설정 (wolf 2x5) 에서 인자 하나씩 (결정적, 빈 카드 1-6, 비측정).
# 모든 판 --hgt (정답·우리 높이) + 장벽 판 반복별 보폭 k (fem.jsonl bk). 표 marks/cm6_table.md, 끝 표식 marks/cm6.END
C=$1; D=/dev/shm/dkta/impwt/$C; RC=r2:storage/result/anchorflow/lbours/fem_cm/code
export PATH=$HOME/bin:$PATH
if [ ! -f $D/.done ]; then
  mkdir -p /dev/shm/dkta/impwt; rm -rf $D.tmp
  rclone copyto $RC/imp_$C.bundle $D.bundle -q && git clone -q $D.bundle $D.tmp && git -C $D.tmp checkout -q $C && mv $D.tmp $D && echo bundle
  echo $C > $D/.done
fi
cd $D || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
W=/dev/shm/dkta/cm6; mkdir -p $W; MR=r2:storage/result/anchorflow/lbours/fem_cm/marks; RR=r2:storage/result/anchorflow/lbours/fem_cm/cm6
G="--ga hht --rho 0.8 --K 2 --newton 2"; T="--tolN 1e-2 --tolP 1e-1"; B="$G --cg 5 $T --det --hgt"
L=$W/list.txt; cat > $L <<EOT
r wolf $B --dump
stk wolf $B --no=--fl_bar --no=--fl_bst --dump
nost wolf $B --fl_nost --dump
bst0 wolf $B --no=--fl_bst
bst2 wolf $B --fl_bst 2
oct wolf $B --no=--gq_oct
u wolf $B --no=--ga_y0ut
jac wolf $B --mg_levels 1
hs wolf $B --no=--hspec --no=--ncomp
c20 wolf $B --cg 20
cv wolf --ga hht --rho 0.8 --K 2 --newton 8 --cg 40 --tolN 1e-4 --tolP 1e-2 --det --hgt --dump
cvs wolf --ga hht --rho 0.8 --K 2 --newton 8 --cg 40 --tolN 1e-4 --tolP 1e-2 --det --hgt --no=--fl_bar --no=--fl_bst
k8 wolf --ga hht --rho 0.8 --K 8 --newton 2 --cg 5 $T --det --hgt
r7k3 wolf $B --rho 0.7 --K 3
old wolf --ga hht --rho 0.7 --ga_y0ut --K 3 --newton 2 --cg 5 $T --det --hgt --noba --dump
oldbar wolf --ga hht --rho 0.7 --ga_y0ut --K 3 --newton 2 --cg 5 $T --det --hgt --noba --fl_bar --fl_bst 1
oldnost wolf --ga hht --rho 0.7 --ga_y0ut --K 3 --newton 2 --cg 5 $T --det --hgt --noba --fl_nost
EOT
rclone copy $L $RR -q
CM_SCN=exe/fem_cm6_scn.sh CM_CARDS="1 2 3 4 5 6" CM_PAR=6 CM_GAP=5 CM_TMO=1200 bash exe/fem_cm_pool.sh cm6 $L > $W/pool.log 2>&1
rclone copyto $W/pool.log $MR/cm6_pool.log -q
R=/dev/shm/dkta/lbours/fem_cm/cm6
for v in r stk nost cv old; do [ -f $R/${v}_wolf/pred.pt ] && rclone copy $R/${v}_wolf/pred.pt $RR/${v}_wolf -q; done
$PY exe/fem_cm6_sum.py $R > $W/cm6_table.md 2>&1
rclone copyto $W/cm6_table.md $MR/cm6_table.md -q
date '+%F %T' > $W/cm6.END; rclone copyto $W/cm6.END $MR/cm6.END -q
echo "[cm6 끝] $(date '+%F %T')"
