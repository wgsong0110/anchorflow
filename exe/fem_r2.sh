#!/bin/bash
# 절 R 2 차: 상자 판 이동 제한 연기 시험 -> 1 차 (fem_r1.sh) 카드 7 장 모두 끝날 때까지 대기 -> GPU 0~6 동시에 (비측정) -> 표·주기.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
LB="--mode lag --opt --mass tet --no_dump --floor vel --rstat --frames 3 --N 96"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
FB="--mode lag --opt --mass tet --no_dump --dump_pred --data $FD --tag F_pg_v0 --frames 3 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --N 640"
S="bash exe/fem_ex_run.sh"
EX_NOWAIT=1 EX_ROOT=fem_r_lego $S r_smoke_box 0 $LB --lim 0.1 --lim_mode box < /dev/null > /dev/null 2>&1
EX_NOWAIT=1 EX_ROOT=fem_r_ficus $S r_smoke_f_box 0 $FB --fb 1 --lim 0.1 --lim_mode box < /dev/null > /dev/null 2>&1
bad=0
for d in $W/lbours/fem_r_lego/r_smoke_box $W/lbours/fem_r_ficus/r_smoke_f_box; do
  if [ -f $d/summary.json ]; then echo "[연기 OK] $(basename $d) $(grep -a '롤아웃\]' $d/v_fem.log | cut -c1-150)"; grep -a "프레임 3/" $d/v_fem.log | cut -c1-400; else echo "[연기 실패] $(basename $d)"; tail -15 $d/v_fem.log; bad=1; fi
done
[ $bad -eq 0 ] || { echo R2_SMOKE_FAIL; exit 1; }
while [ $(ls $W/lbours/fem_r_lego/GPU?_DONE_fem_r_jobs 2>/dev/null | wc -l) -lt 7 ]; do sleep 60; done
echo "[1 차 끝 확인] $(date +%T)"
for G in 0 1 2 3 4 5 6; do bash exe/fem_r_gpu.sh $G exe/fem_r_jobs2.py < /dev/null > $W/fem_r2_g$G.out 2>&1 & done
wait
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_r_ficus/r_f_*/pred.pt > $W/lbours/fem_r_ficus/period2.txt 2>&1
python exe/fem_r_table.py $W/lbours/fem_r_lego/r_l_* > $W/lbours/fem_r_lego/table2.md
python exe/fem_r_table.py $W/lbours/fem_r_ficus/r_f_* > $W/lbours/fem_r_ficus/table2.md
rclone copy $W/lbours/fem_r_lego r2:storage/result/anchorflow/lbours/fem_r_lego -q
rclone copy $W/lbours/fem_r_ficus r2:storage/result/anchorflow/lbours/fem_r_ficus -q
echo R2_DONE $(date +%T)
