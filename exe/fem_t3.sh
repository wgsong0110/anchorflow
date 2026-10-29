#!/bin/bash
# 절 T 3 차: 2 차 (fem_t2.sh) 끝난 뒤 -- 제한 끈 판 중 모든 반복 완주한 가장 작은 N (lego G2P bary N448 5/5, ficus bary N1536 2/2) 을
#   (a) F 덤프 켠 최종 실행 + 3DGS 렌더 (fem_ex_final.sh, GPU0 단독 -- 7 장 500 MiB 아래 대기)  (b) 단독 재측정 3 회 (--no_dump) -> 표·ficus 주기
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
until grep -q T2_DONE $W/fem_t2.out 2>/dev/null; do sleep 60; done
echo "[2 차 끝 확인] $(date +%T)"
LB="--mode lag --opt --p2g bs --bs_mesh occ --floor vel --rstat --bsdiag -1 --g2p bary"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
FB="--mode lag --opt --p2g bs --bs_mesh occ --dump_pred --data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --bsdiag -1 --g2p bary"
fin() {  # fin <형상> <태그> <인자...>
  S=$1; T=$2; shift 2
  for a in 1 2 3; do
    SHAPE=$S EX_ROOT=fem_t_$S bash exe/fem_ex_final.sh ${T}_F$a 0 "$@"
    [ -f $W/lbours/fem_t_$S/${T}_F$a/vj/RENDER_DONE ] && { echo "[최종] $S $T: 시도 $a 에서 완주·렌더"; return; }
    echo "[최종] $S $T: 시도 $a 발산/실패"
  done
}
fin lego f_l_gb_n448 $LB --dump_f32 --N 448
fin ficus f_f_gb_n1536 $FB --dump_f32 --N 1536
for r in 1 2 3; do
  EX_ROOT=fem_t_lego bash exe/fem_ex_run.sh m_l_gb_n448_r$r 0 $LB --no_dump --N 448
  EX_ROOT=fem_t_ficus bash exe/fem_ex_run.sh m_f_gb_n1536_r$r 0 $FB --no_dump --N 1536
done
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_t_ficus/f_f_*/pred.pt $W/lbours/fem_t_ficus/m_f_*/pred.pt > $W/lbours/fem_t_ficus/period3.txt 2>&1
python exe/fem_r_table.py $W/lbours/fem_t_lego/f_l_* $W/lbours/fem_t_lego/m_l_* > $W/lbours/fem_t_lego/table3.md
python exe/fem_r_table.py $W/lbours/fem_t_ficus/f_f_* $W/lbours/fem_t_ficus/m_f_* > $W/lbours/fem_t_ficus/table3.md
python exe/fem_s_cons.py $W/lbours/fem_t_lego/f_l_* $W/lbours/fem_t_lego/m_l_* $W/lbours/fem_t_ficus/f_f_* $W/lbours/fem_t_ficus/m_f_* > $W/lbours/fem_t_lego/cons3.txt 2>&1
rclone copy $W/lbours/fem_t_lego r2:storage/result/anchorflow/lbours/fem_t_lego -q
rclone copy $W/lbours/fem_t_ficus r2:storage/result/anchorflow/lbours/fem_t_ficus -q
echo T3_DONE $(date +%T)
