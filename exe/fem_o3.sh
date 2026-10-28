#!/bin/bash
# 절 O 3 차: (1) 정답 입자 F 특이값 (진단) (2) lego 속도 바닥 N 256·384 (측정, 경계 확인) (3) ficus 최종 판 F 덤프 + 3DGS·ψ 렌더 (lag N768, 같은 코드 경로)
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; R="bash exe/fem_ex_run.sh"
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
mkdir -p $W/lbours/fem_o_lego/gtsig
CUDA_VISIBLE_DEVICES=0 python -u exe/fem_o_gtsig.py < /dev/null > $W/lbours/fem_o_lego/gtsig/gtsig.txt 2>&1; cat $W/lbours/fem_o_lego/gtsig/gtsig.txt
export EX_ROOT=fem_o_lego
for N in 256 384; do $R o_fr_n$N 0 --mode lag --opt --mass tet --no_dump --floor vel --N $N; done
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
PHYS="--data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
SHAPE=ficus EX_ROOT=fem_o_ficus bash exe/fem_ex_final.sh o_fi_final_F 0 --mode lag --opt --mass tet --N 768 --dump_f32 $PHYS
for d in $W/lbours/fem_o_lego/o_fr_n256 $W/lbours/fem_o_lego/o_fr_n384 $W/lbours/fem_o_ficus/o_fi_final_F; do echo "== $(basename $d) $(grep -a "롤아웃\]" $d/v_fem.log | cut -c1-230) $(grep -a "발산\]" $d/v_fem.log|head -1|cut -c1-80)"; done
ls $W/lbours/fem_o_ficus/o_fi_final_F/vj $W/lbours/fem_o_ficus/o_fi_final_F/psi
rclone copy $W/lbours/fem_o_lego r2:storage/result/anchorflow/lbours/fem_o_lego -q
rclone copy $W/lbours/fem_o_ficus r2:storage/result/anchorflow/lbours/fem_o_ficus -q
echo O3_DONE
