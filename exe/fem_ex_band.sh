#!/bin/bash
# 절 L ficus ψ 띠 비교 (fem_rt_psi_band.py) -- 명시적 최종 / 암시적 실시간+수정판정 / 입자 적분점 수정판정
export PYTHONUTF8=1 OMP_NUM_THREADS=4
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
cd "$(dirname "$0")/.." || exit 1
O=$W/lbours/fem_ex_ficus; mkdir -p $O/band
CUDA_VISIBLE_DEVICES=0 python -u exe/fem_rt_psi_band.py --data $W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --t 1,2,3,40,125 --out $O/band/band.json \
   --run "명시적 (lag N768)=$O/f_final_F/roll_fem.pt" --run "암시적 실시간+수정판정=$W/lbours/fem_rt_ficus/rt_tmix_fix/roll_fem.pt" \
   --run "입자 적분점 수정판정(fem3)=$W/lbours/fem3_ficus/run/roll_fem.pt" < /dev/null > $O/band/band.log 2>&1
tail -3 $O/band/band.log | cut -c1-300
rclone copy $O r2:storage/result/anchorflow/lbours/fem_ex_ficus -q
rclone copy $W/lbours/fem_ex_lego/l_final_F r2:storage/result/anchorflow/lbours/fem_ex_lego/l_final_F -q
echo BAND_DONE
