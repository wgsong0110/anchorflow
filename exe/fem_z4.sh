#!/bin/bash
# 절 Z 4 차: 단독 측정 (GPU0, 노드 7 장 500 MiB 아래 대기 -- fem_ex_run.sh) + 최종 F 덤프 3DGS 렌더 (fem_ex_final.sh, 발산하면 최대 3 번).
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
LB="--mode lag --opt --mass tet --floor vel --rstat"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
FB="--mode lag --opt --mass tet --dump_pred --data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat"
T="bash exe/fem_ex_run.sh"
for r in 1 2 3; do
  EX_ROOT=fem_z_lego $T t_l_d03_n192_r$r 0 $LB --no_dump --lim 0.3 --lim_mode box --N 192 < /dev/null > /dev/null 2>&1
  EX_ROOT=fem_z_lego $T t_l_c2_n192_r$r 0 $LB --no_dump --lim_sr 2 --N 192 < /dev/null > /dev/null 2>&1
  EX_ROOT=fem_z_lego $T t_l_c5d03_n192_r$r 0 $LB --no_dump --lim_sr 5 --lim 0.3 --N 192 < /dev/null > /dev/null 2>&1
  EX_ROOT=fem_z_lego $T t_l_c2_n256_r$r 0 $LB --no_dump --lim_sr 2 --N 256 < /dev/null > /dev/null 2>&1
  EX_ROOT=fem_z_lego $T t_l_c5_n256_r$r 0 $LB --no_dump --lim_sr 5 --N 256 < /dev/null > /dev/null 2>&1
done
for r in 1 2; do
  EX_ROOT=fem_z_ficus $T t_f_c2_n640_r$r 0 $FB --no_dump --lim_sr 2 --N 640 < /dev/null > /dev/null 2>&1
  EX_ROOT=fem_z_ficus $T t_f_sb1c2_n640_r$r 0 $FB --no_dump --sb 1 --sb_c 2 --N 640 < /dev/null > /dev/null 2>&1
done
python exe/fem_z_table.py $W/lbours/fem_z_lego/t_l_* > $W/lbours/fem_z_lego/table4t.md
python exe/fem_z_table.py $W/lbours/fem_z_ficus/t_f_* > $W/lbours/fem_z_ficus/table4t.md
rclone copy $W/lbours/fem_z_lego r2:storage/result/anchorflow/lbours/fem_z_lego -q
rclone copy $W/lbours/fem_z_ficus r2:storage/result/anchorflow/lbours/fem_z_ficus -q
echo "[단독 측정 끝] $(date +%T)"
fin() {  # fin <형상> <태그> <인자...>
  S=$1; TG=$2; shift 2
  for a in 1 2 3; do
    SHAPE=$S EX_ROOT=fem_z_$S bash exe/fem_ex_final.sh ${TG}_F$a 0 "$@" < /dev/null > /dev/null 2>&1
    [ -f $W/lbours/fem_z_$S/${TG}_F$a/vj/RENDER_DONE ] && { echo "[최종] $S $TG: 시도 $a 에서 완주·렌더"; return; }
    echo "[최종] $S $TG: 시도 $a 발산/실패 $(cat $W/lbours/fem_z_$S/${TG}_F$a/err_fem.txt 2>/dev/null | tail -1 | cut -c1-120)"
  done
}
fin lego f_l_c5_n256 $LB --dump_f32 --lim_sr 5 --N 256
fin lego f_l_c2_n192 $LB --dump_f32 --lim_sr 2 --N 192
fin ficus f_f_c2_n640 $FB --dump_f32 --lim_sr 2 --N 640
fin ficus f_f_sb1c2_n640 $FB --dump_f32 --sb 1 --sb_c 2 --N 640
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_z_ficus/f_f_*/pred.pt > $W/lbours/fem_z_ficus/period4.txt 2>&1
python exe/fem_z_table.py $W/lbours/fem_z_lego/f_l_* > $W/lbours/fem_z_lego/table4f.md
python exe/fem_z_table.py $W/lbours/fem_z_ficus/f_f_* > $W/lbours/fem_z_ficus/table4f.md
rclone copy $W/lbours/fem_z_lego r2:storage/result/anchorflow/lbours/fem_z_lego -q
rclone copy $W/lbours/fem_z_ficus r2:storage/result/anchorflow/lbours/fem_z_ficus -q
echo Z4_DONE $(date +%T)
