#!/bin/bash
# 절 P 3 차: ficus 재평균 판 중 목표에 가장 가까운 것 (N768 K256 full: 17.25 ms·1.473 %) F 덤프 f32 + 3DGS·ψ 렌더 + 주기
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline)"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
PHYS="--data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
SHAPE=ficus EX_ROOT=fem_p_ficus bash exe/fem_ex_final.sh p_fi_final256_F 0 --mode lag --opt --mass tet --N 768 --rep 256 --rep_mode full --dump_f32 --dump_pred $PHYS
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_p_ficus/p_fi_final256_F/pred.pt
d=$W/lbours/fem_p_ficus/p_fi_final256_F; echo "== $(grep -a "롤아웃\]" $d/v_fem.log | cut -c1-230)"; ls $d/vj $d/psi
rclone copy $W/lbours/fem_p_ficus r2:storage/result/anchorflow/lbours/fem_p_ficus -q
echo P3_DONE
