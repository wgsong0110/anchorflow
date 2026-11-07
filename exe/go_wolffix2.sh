#!/bin/bash
# 절 CM27 둘째 진입점: wolf 보충 판 g_dks (--ksplit 3 + (d)), 영상 렌더 (d_locube·e_all), 회귀 두 판 (d = --y0loc 3 --ue be --uf_res, dks = --ksplit 3 + d), 표. 사용: bash go_wolffix2.sh <커밋>
export PATH=/home/dkta/bin:$PATH
H=${1:?커밋}
R=r2:storage/result/anchorflow/lbours/wolffix
O=/dev/shm/dkta/lbours/wolffix; CB=/tmp/dkta/wolffix_code; C=$CB/$H
mkdir -p $O $CB /dev/shm/dkta/cache /tmp/dkta; rm -f $O/wolffix2.END
L=$O/go2.log; say() { echo "[go2 $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go2.log -q; }
say "시작 $(hostname) 커밋 $H"
[ -s $CB/wolffix_$H.bundle ] || rclone copyto $R/code/wolffix_$H.bundle $CB/wolffix_$H.bundle -q
[ -d $C/.git ] || git clone -q -b k1new $CB/wolffix_$H.bundle $C
cd $C && git checkout -q $H || { say "코드 실패"; echo "코드 실패" | rclone rcat $R/wolffix2.END; exit 1; }
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_wolffix XDG_CACHE_HOME=/dev/shm/dkta/cache
PY=/home/dkta/.conda/envs/af/bin/python
WDP=/tmp/dkta/ag_data/wolf.pt; [ -s $WDP ] || WDP=/dev/shm/dkta/ag_data/wolf.pt
( bash exe/stab_wolffix_vid.sh $O $R d_locube e_all < /dev/null > $O/stab_wolffix_vid.out 2>&1; say "영상 끝" ) &
( WF_EXTRA="f_ks f_ksloc f_ksphi f_ksall g_dks" bash exe/stab_wolffix.sh $O $R < /dev/null > $O/stab_wolffix2.out 2>&1; say "wolf g_dks 끝" ) &
mkdir -p $O/reg
( bash exe/stab_wolffix_reg.sh $O/reg $R/reg d --y0loc 3 --ue be --uf_res < /dev/null > $O/reg/stab_d.out 2>&1; say "회귀 d 끝"
  bash exe/stab_wolffix_reg.sh $O/reg $R/reg dks --ksplit 3 --y0loc 3 --ue be --uf_res < /dev/null > $O/reg/stab_dks.out 2>&1; say "회귀 dks 끝" ) &      # 차례로 (off 판 이름 공유)
wait
$PY exe/wolffix_tab.py $O $WDP > $O/wolffix_table.md 2> $O/wolffix_table.err
$PY exe/wolffix_regtab.py $O/reg d dks > $O/reg/wolffix_reg_table.md 2> $O/reg/wolffix_reg_table.err
rclone copy $O $R --exclude "wolffix*.END" -q --transfers 8
D=$(rclone check $O $R --one-way --size-only --exclude "wolffix*.END" 2>&1 | grep -c ERROR)
say "R2 확인 차이 $D"
echo "끝 $(date '+%F %T') R2 차이 $D" > $O/wolffix2.END; rclone copyto $O/wolffix2.END $R/wolffix2.END -q
say "끝"
