#!/bin/bash
# 절 P 2 차: (1) ficus 같은 코드 경로 추가 (vel K 1·2·4·8·16, full K 256 @ N768; 측정·주기)
#            (2) lego 목표에 가장 가까운 판 (N192 K16 vel: 15.2 ms·13.3 %) F 덤프 + 3DGS 렌더
#            (3) ficus 재평균 판 중 완주한 것 (N768 K128 full) F 덤프 f32 + 3DGS·ψ 렌더 + 주기
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; R="bash exe/fem_ex_run.sh"
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline)"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
PHYS="--data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
FB="--mode lag --opt --mass tet --no_dump --dump_pred $PHYS"
export EX_ROOT=fem_p_ficus
for K in 16 8 4 2 1; do $R p_f_n768_k${K}_vel 0 $FB --N 768 --rep $K --rep_mode vel; done
$R p_f_n768_k256_full 0 $FB --N 768 --rep 256 --rep_mode full
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_p_ficus/p_f_*/pred.pt | tee $W/lbours/fem_p_ficus/period2.txt
python exe/fem_p_table.py $W/lbours/fem_p_ficus/p_f_* | tee $W/lbours/fem_p_ficus/table2.md
rclone copy $W/lbours/fem_p_ficus r2:storage/result/anchorflow/lbours/fem_p_ficus -q
SHAPE=lego EX_ROOT=fem_p_lego bash exe/fem_ex_final.sh p_lego_final_F 0 --mode lag --opt --mass tet --floor vel --N 192 --rep 16 --rep_mode vel
SHAPE=ficus EX_ROOT=fem_p_ficus bash exe/fem_ex_final.sh p_fi_final_F 0 --mode lag --opt --mass tet --N 768 --rep 128 --rep_mode full --dump_f32 --dump_pred $PHYS
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_p_ficus/p_fi_final_F/pred.pt
for d in $W/lbours/fem_p_lego/p_lego_final_F $W/lbours/fem_p_ficus/p_fi_final_F; do echo "== $(basename $d) $(grep -a "롤아웃\]" $d/v_fem.log | cut -c1-230)"; ls $d/vj $d/psi 2>/dev/null; done
rclone copy $W/lbours/fem_p_lego r2:storage/result/anchorflow/lbours/fem_p_lego -q
rclone copy $W/lbours/fem_p_ficus r2:storage/result/anchorflow/lbours/fem_p_ficus -q
echo P2_DONE
