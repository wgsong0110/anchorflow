#!/bin/bash
# 절 S 1 차: 연기 시험 (3 프레임, 후보마다 + 진단) -> 통과하면 GPU 0~6 에 나눠 동시에 (카드마다 fem_s_gpu.sh, 비측정) -> 표·ficus 주기.  속도는 2 차 (단독 재측정) 값을 쓴다.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
mkdir -p $W/lbours/fem_s_lego $W/lbours/fem_s_ficus; rm -f $W/lbours/fem_s_lego/GPU?_DONE_*
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader
LB="--mode lag --opt --p2g bs --no_dump --floor vel --rstat --bsdiag 1 --frames 3 --N 192"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
FB="--mode lag --opt --p2g bs --no_dump --dump_pred --data $FD --tag F_pg_v0 --frames 3 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --bsdiag 1 --N 640"
S="bash exe/fem_ex_run.sh"
export EX_NOWAIT=1
( EX_ROOT=fem_s_lego $S s_smoke_gb 0 $LB --g2p bary ) > /dev/null 2>&1 &
( EX_ROOT=fem_s_lego $S s_smoke_gs 1 $LB --g2p bs ) > /dev/null 2>&1 &
( EX_ROOT=fem_s_lego $S s_smoke_gsL 2 $LB --g2p bs --lim 0.3 --lim_mode box ) > /dev/null 2>&1 &
( EX_ROOT=fem_s_lego $S s_smoke_gs_ng 3 $LB --g2p bs --no_graph ) > /dev/null 2>&1 &
( EX_ROOT=fem_s_ficus $S s_smoke_f_gb 4 $FB --g2p bary ) > /dev/null 2>&1 &
( EX_ROOT=fem_s_ficus $S s_smoke_f_gsL 5 $FB --g2p bs --lim 0.3 --lim_mode box ) > /dev/null 2>&1 &
( EX_ROOT=fem_s_lego $S s_smoke_old 6 --mode lag --opt --mass tet --no_dump --floor vel --rstat --frames 3 --N 192 ) > /dev/null 2>&1 &
wait
unset EX_NOWAIT
bad=0
for d in $W/lbours/fem_s_lego/s_smoke_* $W/lbours/fem_s_ficus/s_smoke_*; do
  if [ -f $d/summary.json ]; then echo "[연기 OK] $(basename $d) $(grep -a '롤아웃\]' $d/v_fem.log | cut -c1-150)"; grep -a "프레임 [123]/\|S 진단" $d/v_fem.log | cut -c1-700; else echo "[연기 실패] $(basename $d)"; tail -25 $d/v_fem.log; bad=1; fi
done
rclone copy $W/lbours/fem_s_lego r2:storage/result/anchorflow/lbours/fem_s_lego -q
rclone copy $W/lbours/fem_s_ficus r2:storage/result/anchorflow/lbours/fem_s_ficus -q
[ $bad -eq 0 ] || { echo S1_SMOKE_FAIL; exit 1; }
echo S1_SMOKE_OK $(date +%T)
for G in 0 1 2 3 4 5 6; do bash exe/fem_s_gpu.sh $G < /dev/null > $W/fem_s1_g$G.out 2>&1 & done
wait
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_s_ficus/s_f_*/pred.pt > $W/lbours/fem_s_ficus/period1.txt 2>&1
python exe/fem_r_table.py $W/lbours/fem_s_lego/s_l_* > $W/lbours/fem_s_lego/table1.md
python exe/fem_r_table.py $W/lbours/fem_s_ficus/s_f_* > $W/lbours/fem_s_ficus/table1.md
rclone copy $W/lbours/fem_s_lego r2:storage/result/anchorflow/lbours/fem_s_lego -q
rclone copy $W/lbours/fem_s_ficus r2:storage/result/anchorflow/lbours/fem_s_ficus -q
echo S1_DONE $(date +%T)
