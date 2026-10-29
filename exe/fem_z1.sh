#!/bin/bash
# 절 Z 1 차 (사용자 10-29 "A_T 특이값 비율 σ1/σ3 로 (1) 클램프 (2) 로그 장벽"): 정답 F 비율 분포 + 검산 + 연기 시험 (GPU 4·5·6, 다른 진단 잡 카드 0~3 피함, 비측정).
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
OL=$W/lbours/fem_z_lego; OF=$W/lbours/fem_z_ficus; mkdir -p $OL $OF
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
( CUDA_VISIBLE_DEVICES=4 python -u exe/fem_z_gtsr.py --data $W/one/L000/L000.pt --frames 80 --out $OL/gt_sr.json > $OL/gt_sr.log 2>&1
  CUDA_VISIBLE_DEVICES=4 python -u exe/fem_z_gtsr.py --data $FD --frames 125 --out $OF/gt_sr.json > $OF/gt_sr.log 2>&1 ) < /dev/null &
( for cc in 1.5 2 3 5; do CUDA_VISIBLE_DEVICES=5 python -u exe/fem_z_check.py --c $cc --T 1000; done > $OL/check.log 2>&1 ) < /dev/null &
export EX_NOWAIT=1
LB="--mode lag --opt --mass tet --no_dump --floor vel --rstat --frames 3 --N 192"
FB="--mode lag --opt --mass tet --no_dump --dump_pred --data $FD --tag F_pg_v0 --frames 3 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --N 640"
( EX_ROOT=fem_z_lego bash exe/fem_ex_run.sh z_smoke_cl 6 $LB --lim_sr 2 --lim_sr_chk
  EX_ROOT=fem_z_lego bash exe/fem_ex_run.sh z_smoke_cld 6 $LB --lim_sr 2 --lim 0.3 --lim_sr_chk
  EX_ROOT=fem_z_lego bash exe/fem_ex_run.sh z_smoke_sb 6 $LB --sb 1 --sb_c 2
  EX_ROOT=fem_z_lego bash exe/fem_ex_run.sh z_smoke_ref 6 $LB
  EX_ROOT=fem_z_ficus bash exe/fem_ex_run.sh z_smoke_f_cl 6 $FB --lim_sr 2 --lim_sr_chk
  EX_ROOT=fem_z_ficus bash exe/fem_ex_run.sh z_smoke_f_sb 6 $FB --sb 1 --sb_c 2 ) < /dev/null > /dev/null 2>&1 &
wait
for d in $OL/z_smoke_* $OF/z_smoke_*; do echo "== $(basename $d)"; grep -a "\[명령\]\|프레임 [0-9]*/\|롤아웃\]\|Error\|error" $d/v_fem.log | cut -c1-600 | tail -6; done
tail -2 $OL/gt_sr.log $OF/gt_sr.log
cat $OL/check.log
rclone copy $OL r2:storage/result/anchorflow/lbours/fem_z_lego -q
rclone copy $OF r2:storage/result/anchorflow/lbours/fem_z_ficus -q
echo Z1_DONE $(date +%T)
