#!/bin/bash
# 절 L 7 차: lego 명시적 최종 판 (sub 융합, 사면체 등분 질량, N 96, 정렬 끔) F 덤프·렌더 + ficus (경계 N 640, 반복, 기준 판, 최종 F 덤프 f32·렌더·ψ 색·ψ 띠)
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; R="bash exe/fem_ex_run.sh"
SHAPE=lego bash exe/fem_ex_final.sh l_final_F 0 --mode sub --N 96 --mass tet --opt --near32 --no_sort
echo CHAIN7_LEGO_DONE
export EX_ROOT=fem_ex_ficus
PHYS="--data $W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
$R f_lag_tm_n640 0 --mode lag --N 640 --opt --mass tet --no_dump $PHYS
$R f_lag_tm_n768_r2 0 --mode lag --N 768 --opt --mass tet --no_dump $PHYS
$R f_lag_tm_n768_nog 0 --mode lag --N 768 --opt --mass tet --no_dump --no_graph $PHYS
$R f_lag_tm_n768_base 0 --mode lag --N 768 --mass tet --no_dump $PHYS
SHAPE=ficus bash exe/fem_ex_final.sh f_final_F 0 --mode lag --N 768 --opt --mass tet --dump_f32 $PHYS
O=$W/lbours/fem_ex_ficus; mkdir -p $O/band
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af
CUDA_VISIBLE_DEVICES=0 python -u exe/fem_rt_psi_band.py --data $W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --t 1,2,3,40,125 --out $O/band/band.json \
   --run "명시적 (lag N768)=$O/f_final_F/roll_fem.pt" --run "암시적 실시간+수정판정=$W/lbours/fem_rt_ficus/rt_tmix_fix/roll_fem.pt" \
   --run "입자 적분점 수정판정(fem3)=$W/lbours/fem3_ficus/run/roll_fem.pt" < /dev/null > $O/band/band.log 2>&1
tail -5 $O/band/band.log | cut -c1-800
rclone copy $O r2:storage/result/anchorflow/lbours/fem_ex_ficus -q
echo CHAIN7_DONE
