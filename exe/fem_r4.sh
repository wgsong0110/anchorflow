#!/bin/bash
# 절 R 4 차: 대표 판 F 덤프 (단독 측정, GPU0) + 3DGS 알파블렌딩 렌더 (fem_ex_final.sh). lego 는 같은 설정도 발산 여부가 갈려 발산하면 다시 (최대 3 번, _F1~_F3).
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
LB="--mode lag --opt --mass tet --floor vel --rstat"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
FB="--mode lag --opt --mass tet --dump_f32 --dump_pred --data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat"
fin() {  # fin <형상> <태그> <인자...>
  S=$1; T=$2; shift 2
  for a in 1 2 3; do
    SHAPE=$S EX_ROOT=fem_r_$S bash exe/fem_ex_final.sh ${T}_F$a 0 "$@"
    [ -f $W/lbours/fem_r_$S/${T}_F$a/vj/RENDER_DONE ] && { echo "[최종] $S $T: 시도 $a 에서 완주·렌더"; return; }
    echo "[최종] $S $T: 시도 $a 발산/실패"
  done
}
fin lego f_l_b03_n192 $LB --dump_f32 --lim 0.3 --lim_mode box --N 192
fin lego f_l_b05_n256 $LB --dump_f32 --lim 0.5 --lim_mode box --N 256
fin ficus f_f_b03_n640 $FB --lim 0.3 --lim_mode box --N 640
fin ficus f_f_fX1_n640 $FB --fb 1 --N 640
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_r_ficus/f_f_*/pred.pt > $W/lbours/fem_r_ficus/period4.txt 2>&1
python exe/fem_r_table.py $W/lbours/fem_r_lego/f_l_* > $W/lbours/fem_r_lego/table4.md
python exe/fem_r_table.py $W/lbours/fem_r_ficus/f_f_* > $W/lbours/fem_r_ficus/table4.md
rclone copy $W/lbours/fem_r_lego r2:storage/result/anchorflow/lbours/fem_r_lego -q
rclone copy $W/lbours/fem_r_ficus r2:storage/result/anchorflow/lbours/fem_r_ficus -q
echo R4_DONE $(date +%T)
