#!/bin/bash
# 절 AA 1 차 (사용자 10-30 "셀 평균 ψ 의 근사로"): 연기 시험 + 검산 (비측정, GPU 1·2 -- GPU 0 은 같은 잡의 절 Z 단독 측정 몫).
#  lego 3 프레임 (A = 프레임 끝/I 셀 에너지 세 가지, 힘 km·cf vs float64 자동 미분·중앙 차분), lego 40 프레임 이동 제한 0.3 (프레임 10·20·30·40 검산), ficus 3 프레임.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1 EX_NOWAIT=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
OL=$W/lbours/fem_aa_lego; OF=$W/lbours/fem_aa_ficus; mkdir -p $OL $OF
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
LB="--mode lag --opt --mass tet --no_dump --floor vel --rstat --cq"
FB="--mode lag --opt --mass tet --no_dump --dump_pred --data $FD --tag F_pg_v0 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --cq"
T="bash exe/fem_ex_run.sh"
( EX_ROOT=fem_aa_lego $T aa_smoke_l 1 $LB --frames 3 --N 192 --cqchk 1,2,3 --fspread 1
  EX_ROOT=fem_aa_lego $T aa_smoke_lcf 1 $LB --cq_h cf --frames 3 --N 192
  EX_ROOT=fem_aa_ficus $T aa_smoke_f 1 $FB --frames 3 --N 640 --cqchk 1,3 --fspread 1 ) < /dev/null > /dev/null 2>&1 &
( EX_ROOT=fem_aa_lego $T aa_chk_l40 2 $LB --frames 40 --N 192 --lim 0.3 --lim_mode box --cqchk 10,20,30,40 --fspread 10 ) < /dev/null > /dev/null 2>&1 &
wait
for d in $OL/aa_* $OF/aa_*; do echo "== $(basename $d)"; grep -a "\[명령\]\|프레임 [0-9]*/\|AA 검산\|롤아웃\]\|Error\|error\|Traceback" $d/v_fem.log | cut -c1-900 | tail -12; done
rclone copy $OL r2:storage/result/anchorflow/lbours/fem_aa_lego -q
rclone copy $OF r2:storage/result/anchorflow/lbours/fem_aa_ficus -q
echo AA1_DONE $(date +%T)
