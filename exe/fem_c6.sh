#!/bin/bash
# 절 C 6 차: 최종 판 (F 덤프 f32, 단독 측정 GPU0 flock + 3DGS 렌더 GPU5, 발산하면 최대 3 번) -- /home 여유 12 GB 이상일 때까지 기다린 뒤.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
while [ "$(df --output=avail -B1G /home | tail -1 | tr -d ' ')" -lt 12 ]; do echo "[디스크 대기] $(df -h /home | tail -1) $(date +%T)"; sleep 300; done
echo "[디스크] $(df -h /home | tail -1)"
LB="--mode lag --opt --mass tet --floor vel --rstat"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
FB="--mode lag --opt --mass tet --dump_pred --data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat"
F2="--cfuse --cfuse_mode 2"
fin() {  # fin <형상> <태그> <인자...>
  S=$1; TG=$2; shift 2
  for a in 1 2 3; do
    RG=5 SHAPE=$S EX_ROOT=fem_c_$S bash exe/fem_c_final.sh ${TG}_F$a 0 "$@" < /dev/null > /dev/null 2>&1
    [ -f $W/lbours/fem_c_$S/${TG}_F$a/vj/RENDER_DONE ] && { echo "[최종] $S $TG: 시도 $a 에서 완주·렌더 $(cat $W/lbours/fem_c_$S/${TG}_F$a/err_fem.txt | tail -1 | cut -c1-160)"; return; }
    echo "[최종] $S $TG: 시도 $a 발산/실패 $(cat $W/lbours/fem_c_$S/${TG}_F$a/err_fem.txt 2>/dev/null | tail -1 | cut -c1-160)"
  done
}
fin lego f_l_c5_n256_fu2 $LB --dump_f32 --lim_sr 5 --N 256 $F2
fin ficus f_f_c2_n640_fu2 $FB --dump_f32 --lim_sr 2 --N 640 $F2
fin lego f_l_c2_n192_fu2 $LB --dump_f32 --lim_sr 2 --N 192 $F2
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_c_ficus/f_f_*/pred.pt > $W/lbours/fem_c_ficus/period6.txt 2>&1
python exe/fem_c_table.py $W/lbours/fem_c_lego/f_l_* > $W/lbours/fem_c_lego/table6f.md 2>&1
python exe/fem_c_table.py $W/lbours/fem_c_ficus/f_f_* > $W/lbours/fem_c_ficus/table6f.md 2>&1
cat $W/lbours/fem_c_lego/table6f.md $W/lbours/fem_c_ficus/table6f.md $W/lbours/fem_c_ficus/period6.txt
rclone copy $W/lbours/fem_c_lego r2:storage/result/anchorflow/lbours/fem_c_lego -q
rclone copy $W/lbours/fem_c_ficus r2:storage/result/anchorflow/lbours/fem_c_ficus -q
echo C6_DONE $(date +%T)
