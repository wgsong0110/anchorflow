#!/bin/bash
# 절 Q 3 차: 후보별 최고 판 단독 재측정 (GPU0, 노드 7 장 모두 빌 때까지 대기 -- fem_ex_run.sh 기본), 판마다 2 번 (재현성: 원자 합 순서로 안정 여부가 갈린다)
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; R="bash exe/fem_ex_run.sh"
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline)"
LB="--mode lag --opt --no_dump --floor vel --mass tet"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
FB="--mode lag --opt --no_dump --dump_pred --data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --mass tet"
for r in 1 2; do
  export EX_ROOT=fem_q_lego
  $R s_l_a4_vel_n240_k48_r$r 0 $LB --rep_w tet4 --N 240 --rep 48 --rep_mode vel
  $R s_l_a4_vel_n256_k48_r$r 0 $LB --rep_w tet4 --N 256 --rep 48 --rep_mode vel
  $R s_l_a4_vel_n192_k48_r$r 0 $LB --rep_w tet4 --N 192 --rep 48 --rep_mode vel
  $R s_l_a4_full_n240_k32_r$r 0 $LB --rep_w tet4 --N 240 --rep 32 --rep_mode full
  $R s_l_aL_vel_n192_k32_r$r 0 $LB --rep_w lam --N 192 --rep 32 --rep_mode vel
  $R s_l_aL_vel_n256_k32_r$r 0 $LB --rep_w lam --N 256 --rep 32 --rep_mode vel
  $R s_l_std_vel_n192_k16_r$r 0 $LB --N 192 --rep 16 --rep_mode vel
  $R s_l_std_vel_n256_k64_r$r 0 $LB --N 256 --rep 64 --rep_mode vel
  $R s_l_std_vel_n256_k48_r$r 0 $LB --N 256 --rep 48 --rep_mode vel
  $R s_l_std_vel_n240_k32_r$r 0 $LB --N 240 --rep 32 --rep_mode vel
  export EX_ROOT=fem_q_ficus
  $R s_f_a4_vel_n640_k256_r$r 0 $FB --rep_w tet4 --N 640 --rep 256 --rep_mode vel
  $R s_f_aL_vel_n640_k256_r$r 0 $FB --rep_w lam --N 640 --rep 256 --rep_mode vel
  $R s_f_aL_full_n640_k256_r$r 0 $FB --rep_w lam --N 640 --rep 256 --rep_mode full
  $R s_f_b_n640_k0_r$r 0 $FB --affine --N 640
  $R s_f_b_full_n640_k128_r$r 0 $FB --affine --N 640 --rep 128 --rep_mode full
  $R s_f_bL_n640_k0_r$r 0 $FB --affine --rep_w lam --N 640
  $R s_f_bL_vel_n768_k128_r$r 0 $FB --affine --rep_w lam --N 768 --rep 128 --rep_mode vel
  $R s_f_std_n768_k0_r$r 0 $FB --N 768
  $R s_f_std_n640_k0_r$r 0 $FB --N 640
  $R s_f_std_vel_n640_k256_r$r 0 $FB --N 640 --rep 256 --rep_mode vel
done
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_q_ficus/s_f_*/pred.pt > $W/lbours/fem_q_ficus/period3.txt 2>&1
python exe/fem_q_table.py $W/lbours/fem_q_lego/s_l_* > $W/lbours/fem_q_lego/table3.md
python exe/fem_q_table.py $W/lbours/fem_q_ficus/s_f_* > $W/lbours/fem_q_ficus/table3.md
rclone copy $W/lbours/fem_q_lego r2:storage/result/anchorflow/lbours/fem_q_lego -q
rclone copy $W/lbours/fem_q_ficus r2:storage/result/anchorflow/lbours/fem_q_ficus -q
echo Q3_DONE
