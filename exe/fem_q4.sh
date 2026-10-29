#!/bin/bash
# 절 Q 4 차: 후보별 대표 판 F 덤프 (단독 측정, GPU0) + 3DGS 알파블렌딩 렌더 (fem_ex_final.sh). lego 는 같은 설정도 원자 합 순서로 발산 여부가 갈려 (3 차 재현 시험)
#   발산하면 같은 설정으로 다시 (최대 3 번, 태그 _F1·_F2·_F3) -- 몇 번째에 완주했는지 기록.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline)"
LB="--mode lag --opt --mass tet --floor vel"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
FB="--mode lag --opt --mass tet --dump_f32 --dump_pred --data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
fin() {  # fin <형상> <태그> <인자...>
  S=$1; T=$2; shift 2
  for a in 1 2 3; do
    SHAPE=$S EX_ROOT=fem_q_$S bash exe/fem_ex_final.sh ${T}_F$a 0 "$@"
    [ -f $W/lbours/fem_q_$S/${T}_F$a/vj/RENDER_DONE ] && { echo "[최종] $S $T: 시도 $a 에서 완주·렌더"; return; }
    echo "[최종] $S $T: 시도 $a 발산/실패"
  done
}
fin lego f_l_aL_vel_n256_k32 $LB --rep_w lam --N 256 --rep 32 --rep_mode vel
fin lego f_l_std_vel_n256_k48 $LB --N 256 --rep 48 --rep_mode vel
fin lego f_l_std_vel_n192_k16 $LB --N 192 --rep 16 --rep_mode vel
fin ficus f_f_aL_vel_n640_k256 $FB --rep_w lam --N 640 --rep 256 --rep_mode vel
fin ficus f_f_b_full_n640_k128 $FB --affine --N 640 --rep 128 --rep_mode full
fin ficus f_f_bL_vel_n768_k128 $FB --affine --rep_w lam --N 768 --rep 128 --rep_mode vel
fin ficus f_f_std_n768_k0 $FB --N 768
fin ficus f_f_std_vel_n640_k256 $FB --N 640 --rep 256 --rep_mode vel
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_q_ficus/f_f_*/pred.pt > $W/lbours/fem_q_ficus/period4.txt 2>&1
python exe/fem_q_table.py $W/lbours/fem_q_lego/f_l_* > $W/lbours/fem_q_lego/table4.md
python exe/fem_q_table.py $W/lbours/fem_q_ficus/f_f_* > $W/lbours/fem_q_ficus/table4.md
rclone copy $W/lbours/fem_q_lego r2:storage/result/anchorflow/lbours/fem_q_lego -q
rclone copy $W/lbours/fem_q_ficus r2:storage/result/anchorflow/lbours/fem_q_ficus -q
echo Q4_DONE
