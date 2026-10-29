#!/bin/bash
# 절 R 1 차: 연기 시험 (3 프레임, 후보마다) -> GPU 0~6 에 나눠 동시에 (카드마다 fem_r_gpu.sh 하나, 비측정) -> 표·ficus 주기.  속도는 2 차 (단독 재측정) 값을 쓴다.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
mkdir -p $W/lbours/fem_r_lego $W/lbours/fem_r_ficus; rm -f $W/lbours/fem_r_lego/GPU?_DONE_*
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader
LB="--mode lag --opt --mass tet --no_dump --floor vel --rstat --frames 3 --N 96"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
FB="--mode lag --opt --mass tet --no_dump --dump_pred --data $FD --tag F_pg_v0 --frames 3 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --N 640"
S="bash exe/fem_ex_run.sh"
export EX_NOWAIT=1
( EX_ROOT=fem_r_lego $S r_smoke_base 0 $LB; EX_ROOT=fem_r_lego $S r_smoke_old 0 --mode lag --opt --mass tet --no_dump --floor vel --frames 3 --N 96 ) > /dev/null 2>&1 &
( EX_ROOT=fem_r_lego $S r_smoke_fX 1 $LB --fb 1; EX_ROOT=fem_r_lego $S r_smoke_fA 1 $LB --fb 1 --fb_on A ) > /dev/null 2>&1 &
( EX_ROOT=fem_r_lego $S r_smoke_l 2 $LB --lim 0.1; EX_ROOT=fem_r_lego $S r_smoke_lk 2 $LB --lim 0.1 --lim_v keep ) > /dev/null 2>&1 &
( EX_ROOT=fem_r_lego $S r_smoke_fl 3 $LB --fb 1 --lim 0.1 ) > /dev/null 2>&1 &
( EX_ROOT=fem_r_ficus $S r_smoke_f_l 4 $FB --fb 1 --lim 0.1 ) > /dev/null 2>&1 &
( EX_ROOT=fem_r_lego $S r_smoke_ldiag 5 --mode lag --opt --mass tet --no_dump --floor vel --rstat --frames 3 --N 96 --lim 0.1 --lagdiag ) > /dev/null 2>&1 &
wait
unset EX_NOWAIT
bad=0
for d in $W/lbours/fem_r_lego/r_smoke_* $W/lbours/fem_r_ficus/r_smoke_*; do
  if [ -f $d/summary.json ]; then echo "[연기 OK] $(basename $d) $(grep -a '롤아웃\]' $d/v_fem.log | cut -c1-150)"; grep -a "프레임 3/" $d/v_fem.log | cut -c1-400; else echo "[연기 실패] $(basename $d)"; tail -15 $d/v_fem.log; bad=1; fi
done
[ $bad -eq 0 ] || { echo R1_SMOKE_FAIL; exit 1; }
for G in 0 1 2 3 4 5 6; do bash exe/fem_r_gpu.sh $G < /dev/null > $W/fem_r1_g$G.out 2>&1 & done
wait
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_r_ficus/r_f_*/pred.pt > $W/lbours/fem_r_ficus/period1.txt 2>&1
python exe/fem_r_table.py $W/lbours/fem_r_lego/r_l_* > $W/lbours/fem_r_lego/table1.md
python exe/fem_r_table.py $W/lbours/fem_r_ficus/r_f_* > $W/lbours/fem_r_ficus/table1.md
rclone copy $W/lbours/fem_r_lego r2:storage/result/anchorflow/lbours/fem_r_lego -q
rclone copy $W/lbours/fem_r_ficus r2:storage/result/anchorflow/lbours/fem_r_ficus -q
echo R1_DONE $(date +%T)
