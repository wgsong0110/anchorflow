#!/bin/bash
# 절 Q 1 차: GPU 0~6 에 나눠 동시에 (카드마다 fem_q_gpu.sh 하나, 비측정) -> 모두 끝나면 ficus 주기·표.  속도는 2 차 (단독 재측정) 값을 쓴다.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline)"
mkdir -p $W/lbours/fem_q_lego $W/lbours/fem_q_ficus; rm -f $W/lbours/fem_q_lego/GPU?_DONE
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader
# 0. 연기 시험 (3 프레임, 비측정, 후보마다 다른 카드) -- 하나라도 실패하면 멈춘다
LB="--mode lag --opt --no_dump --floor vel --frames 3"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
FB="--mode lag --opt --no_dump --dump_pred --data $FD --tag F_pg_v0 --frames 3 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
S="bash exe/fem_ex_run.sh"
( EX_ROOT=fem_q_lego EX_NOWAIT=1 $S q_smoke_a4v 0 $LB --mass tet --rep_w tet4 --N 96 --rep 8 --rep_mode vel; EX_ROOT=fem_q_lego EX_NOWAIT=1 $S q_smoke_a4f 0 $LB --mass tet --rep_w tet4 --N 96 --rep 8 --rep_mode full ) > /dev/null 2>&1 &
( EX_ROOT=fem_q_lego EX_NOWAIT=1 $S q_smoke_aLv 1 $LB --mass tet --rep_w lam --N 96 --rep 8 --rep_mode vel; EX_ROOT=fem_q_lego EX_NOWAIT=1 $S q_smoke_aLf 1 $LB --mass tet --rep_w lam --N 96 --rep 8 --rep_mode full ) > /dev/null 2>&1 &
( EX_ROOT=fem_q_lego EX_NOWAIT=1 $S q_smoke_bf 2 $LB --mass tet --affine --N 96 --rep 8 --rep_mode full; EX_ROOT=fem_q_lego EX_NOWAIT=1 $S q_smoke_bv 2 $LB --mass tet --affine --N 96 --rep 8 --rep_mode vel ) > /dev/null 2>&1 &
( EX_ROOT=fem_q_lego EX_NOWAIT=1 $S q_smoke_bLf 3 $LB --mass tet --affine --rep_w lam --N 96 --rep 8 --rep_mode full; EX_ROOT=fem_q_lego EX_NOWAIT=1 $S q_smoke_bBf 3 $LB --mass bary --affine --N 96 --rep 8 --rep_mode full ) > /dev/null 2>&1 &
( EX_ROOT=fem_q_ficus EX_NOWAIT=1 $S q_smoke_f_a4 4 $FB --mass tet --rep_w tet4 --N 768 --rep 128 --rep_mode full; EX_ROOT=fem_q_ficus EX_NOWAIT=1 $S q_smoke_f_aL 4 $FB --mass tet --rep_w lam --N 768 ) > /dev/null 2>&1 &
( EX_ROOT=fem_q_ficus EX_NOWAIT=1 $S q_smoke_f_b 5 $FB --mass tet --affine --N 768 --rep 128 --rep_mode full; EX_ROOT=fem_q_lego EX_NOWAIT=1 $S q_smoke_std 5 $LB --mass tet --N 96 --rep 8 --rep_mode vel ) > /dev/null 2>&1 &
wait
bad=0
for d in $W/lbours/fem_q_lego/q_smoke_* $W/lbours/fem_q_ficus/q_smoke_*; do
  if [ -f $d/summary.json ]; then echo "[연기 OK] $(basename $d) $(grep -a '롤아웃\]' $d/v_fem.log | cut -c1-150)"; else echo "[연기 실패] $(basename $d)"; tail -15 $d/v_fem.log; bad=1; fi
done
[ $bad -eq 0 ] || { echo Q1_SMOKE_FAIL; exit 1; }
for G in 0 1 2 3 4 5 6; do bash exe/fem_q_gpu.sh $G < /dev/null > $W/fem_q1_g$G.out 2>&1 & done
wait
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_q_ficus/q_f_*/pred.pt > $W/lbours/fem_q_ficus/period1.txt 2>&1
python exe/fem_q_table.py $W/lbours/fem_q_lego/q_l_* > $W/lbours/fem_q_lego/table1.md
python exe/fem_q_table.py $W/lbours/fem_q_ficus/q_f_* > $W/lbours/fem_q_ficus/table1.md
rclone copy $W/lbours/fem_q_lego r2:storage/result/anchorflow/lbours/fem_q_lego -q
rclone copy $W/lbours/fem_q_ficus r2:storage/result/anchorflow/lbours/fem_q_ficus -q
echo Q1_DONE
