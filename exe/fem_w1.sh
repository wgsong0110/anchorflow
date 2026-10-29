#!/bin/bash
# 절 W 1 차: 연기 시험 (3 프레임 + 진단; 카드가 비면) -> 통과하면 GPU 0~6 에 나눠 동시에 스윕 (exe/fem_w_jobs.py, 비측정, 카드마다 비었을 때만) -> 표·ficus 주기·보존
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
mkdir -p $W/lbours/fem_w_lego $W/lbours/fem_w_ficus $W/gpulock_w; rm -f $W/lbours/fem_w_lego/GPU?_DONE_*; rmdir $W/gpulock_w/* 2>/dev/null
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
LB="--mode lag --opt --p2g bs --bs_mesh occ --bs_mass cell --no_dump --floor vel --rstat --bsdiag 1 --frames 3 --N 192"
FB="--mode lag --opt --p2g bs --bs_mesh occ --bs_mass cell --no_dump --dump_pred --data $FD --tag F_pg_v0 --frames 3 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --bsdiag 1 --N 640"
smoke() {  # smoke <G> <루트> <태그> <인자...> -- 카드 잠금·빈 카드 확인 뒤
  G=$1; RT=$2; TG=$3; shift 3
  until mkdir $W/gpulock_w/$G 2>/dev/null; do sleep 20; done
  while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 500 ]; do sleep 30; done
  EX_ROOT=$RT EX_NOWAIT=1 bash exe/fem_ex_run.sh $TG $G "$@" > /dev/null 2>&1
  rmdir $W/gpulock_w/$G
}
smoke 0 fem_w_lego w_smoke_gb $LB --g2p bary &
smoke 1 fem_w_lego w_smoke_gs $LB --g2p bs &
smoke 2 fem_w_lego w_smoke_gsL $LB --g2p bs --lim 0.3 --lim_mode box &
smoke 3 fem_w_lego w_smoke_gs_ng $LB --g2p bs --no_graph &
smoke 4 fem_w_ficus w_smoke_f_gb $FB --g2p bary &
smoke 5 fem_w_ficus w_smoke_f_gs $FB --g2p bs &
smoke 6 fem_w_lego w_smoke_T --mode lag --opt --p2g bs --bs_mesh occ --no_dump --floor vel --rstat --bsdiag 1 --frames 3 --N 192 --g2p bary &
wait
bad=0
for d in $W/lbours/fem_w_lego/w_smoke_* $W/lbours/fem_w_ficus/w_smoke_*; do
  if [ -f $d/summary.json ]; then echo "[연기 OK] $(basename $d) $(grep -a '롤아웃\]' $d/v_fem.log | cut -c1-150)"; grep -a "명시적 FEM\]\|프레임 [123]/\|[SW] 진단\|경고" $d/v_fem.log | cut -c1-1200; else echo "[연기 실패] $(basename $d)"; tail -25 $d/v_fem.log; bad=1; fi
done
python exe/fem_s_cons.py $W/lbours/fem_w_lego/w_smoke_* $W/lbours/fem_w_ficus/w_smoke_*
rclone copy $W/lbours/fem_w_lego r2:storage/result/anchorflow/lbours/fem_w_lego -q
rclone copy $W/lbours/fem_w_ficus r2:storage/result/anchorflow/lbours/fem_w_ficus -q
[ $bad -eq 0 ] || { echo W1_SMOKE_FAIL; exit 1; }
echo W1_SMOKE_OK $(date +%T)
for G in 0 1 2 3 4 5 6; do bash exe/fem_w_gpu.sh $G exe/fem_w_jobs.py $W/lbours/fem_w_lego < /dev/null > $W/fem_w1_g$G.out 2>&1 & done
wait
sleep 20
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_w_ficus/w_f_*/pred.pt > $W/lbours/fem_w_ficus/period1.txt 2>&1
python exe/fem_r_table.py $W/lbours/fem_w_lego/w_l_* > $W/lbours/fem_w_lego/table1.md
python exe/fem_r_table.py $W/lbours/fem_w_ficus/w_f_* > $W/lbours/fem_w_ficus/table1.md
python exe/fem_s_cons.py $W/lbours/fem_w_lego/w_* $W/lbours/fem_w_ficus/w_* > $W/lbours/fem_w_lego/cons1.txt 2>&1
rclone copy $W/lbours/fem_w_lego r2:storage/result/anchorflow/lbours/fem_w_lego -q
rclone copy $W/lbours/fem_w_ficus r2:storage/result/anchorflow/lbours/fem_w_ficus -q
echo W1_DONE $(date +%T)
