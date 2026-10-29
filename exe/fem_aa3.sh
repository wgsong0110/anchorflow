#!/bin/bash
# 절 AA 3 차: 단독 측정 (GPU0, 노드 7 장 500 MiB 아래 대기 -- fem_ex_run.sh) -- 완주한 판 (이동 제한 켬) lego N192·256, ficus N640;
# 헤시안 kmanual (기본) 과 닫힌 식 (같은 값, 속도 비교) -> 최종 F 덤프 3DGS 렌더 (fem_ex_final.sh, 발산하면 최대 3 번)
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
LB="--mode lag --opt --mass tet --floor vel --rstat --cq"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
FB="--mode lag --opt --mass tet --dump_pred --data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --cq"
T="bash exe/fem_ex_run.sh"
for r in 1 2 3; do
  EX_ROOT=fem_aa_lego $T t_l_cqd03_n192_r$r 0 $LB --no_dump --lim 0.3 --lim_mode box --N 192 < /dev/null > /dev/null 2>&1
  EX_ROOT=fem_aa_lego $T t_l_cqd03_n256_r$r 0 $LB --no_dump --lim 0.3 --lim_mode box --N 256 < /dev/null > /dev/null 2>&1
done
for r in 1 2; do
  EX_ROOT=fem_aa_lego $T t_l_cqcfd03_n192_r$r 0 $LB --cq_h cf --no_dump --lim 0.3 --lim_mode box --N 192 < /dev/null > /dev/null 2>&1
  EX_ROOT=fem_aa_ficus $T t_f_cq_n640_r$r 0 $FB --no_dump --N 640 < /dev/null > /dev/null 2>&1
done
EX_ROOT=fem_aa_ficus $T t_f_cqcf_n640_r1 0 $FB --cq_h cf --no_dump --N 640 < /dev/null > /dev/null 2>&1
python exe/fem_aa_table.py $W/lbours/fem_aa_lego/t_l_* > $W/lbours/fem_aa_lego/table3t.md
python exe/fem_aa_table.py $W/lbours/fem_aa_ficus/t_f_* > $W/lbours/fem_aa_ficus/table3t.md
cat $W/lbours/fem_aa_lego/table3t.md $W/lbours/fem_aa_ficus/table3t.md
for d in $W/lbours/fem_aa_*/t_*; do echo "$(basename $d) 시작 $(tr '\n' ' ' < $d/gpu_at_start.txt) 끝 $(tr '\n' ' ' < $d/gpu_at_end.txt)"; done
rclone copy $W/lbours/fem_aa_lego r2:storage/result/anchorflow/lbours/fem_aa_lego -q
rclone copy $W/lbours/fem_aa_ficus r2:storage/result/anchorflow/lbours/fem_aa_ficus -q
echo "[단독 측정 끝] $(date +%T)"
fin() {  # fin <형상> <태그> <인자...>
  S=$1; TG=$2; shift 2
  for a in 1 2 3; do
    SHAPE=$S EX_ROOT=fem_aa_$S bash exe/fem_ex_final.sh ${TG}_F$a 0 "$@" < /dev/null > /dev/null 2>&1
    [ -f $W/lbours/fem_aa_$S/${TG}_F$a/vj/RENDER_DONE ] && { echo "[최종] $S $TG: 시도 $a 에서 완주·렌더"; return; }
    echo "[최종] $S $TG: 시도 $a 발산/실패 $(cat $W/lbours/fem_aa_$S/${TG}_F$a/err_fem.txt 2>/dev/null | tail -1 | cut -c1-120)"
  done
}
fin lego f_l_cqd03_n256 $LB --dump_f32 --lim 0.3 --lim_mode box --N 256
fin ficus f_f_cq_n640 $FB --dump_f32 --N 640
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_aa_ficus/f_f_*/pred.pt $W/lbours/fem_aa_ficus/t_f_*/pred.pt > $W/lbours/fem_aa_ficus/period3.txt 2>&1
python exe/fem_aa_table.py $W/lbours/fem_aa_lego/f_l_* > $W/lbours/fem_aa_lego/table3f.md
python exe/fem_aa_table.py $W/lbours/fem_aa_ficus/f_f_* > $W/lbours/fem_aa_ficus/table3f.md
cat $W/lbours/fem_aa_lego/table3f.md $W/lbours/fem_aa_ficus/table3f.md $W/lbours/fem_aa_ficus/period3.txt
rclone copy $W/lbours/fem_aa_lego r2:storage/result/anchorflow/lbours/fem_aa_lego -q
rclone copy $W/lbours/fem_aa_ficus r2:storage/result/anchorflow/lbours/fem_aa_ficus -q
echo AA3_DONE $(date +%T)
