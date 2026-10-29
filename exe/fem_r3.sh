#!/bin/bash
# 절 R 3 차: (a) Ĵ_min 훑기 GPU 0~6 동시에 (비측정) -> (b) 후보 단독 재측정 (GPU0, 7 장 모두 빌 때까지 대기 -- fem_ex_run.sh 기본), lego 판마다 3 번·ficus 2 번.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; R="bash exe/fem_ex_run.sh"
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
rm -f $W/lbours/fem_r_lego/GPU?_DONE_fem_r_jobs3
for G in 0 1 2 3 4 5 6; do bash exe/fem_r_gpu.sh $G exe/fem_r_jobs3.py < /dev/null > $W/fem_r3_g$G.out 2>&1 & done
wait
python exe/fem_r_table.py $W/lbours/fem_r_lego/r_l_* > $W/lbours/fem_r_lego/table3a.md
echo "[3a 끝] $(date +%T)"
LB="--mode lag --opt --mass tet --no_dump --floor vel --rstat --lim_mode box"
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
FB="--mode lag --opt --mass tet --no_dump --dump_pred --data $FD --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat"
for r in 1 2 3; do
  export EX_ROOT=fem_r_lego
  for N in 160 192 224 256; do $R s_l_b03_n${N}_r$r 0 $LB --lim 0.3 --N $N; done
  $R s_l_base_n192_r$r 0 --mode lag --opt --mass tet --no_dump --floor vel --rstat --N 192
done
for r in 1 2; do
  export EX_ROOT=fem_r_ficus
  $R s_f_base_n640_r$r 0 $FB --N 640
  $R s_f_b03_n640_r$r 0 $FB --lim 0.3 --lim_mode box --N 640
  $R s_f_b01_n640_r$r 0 $FB --lim 0.1 --lim_mode box --N 640
  $R s_f_fX1b01_n640_r$r 0 $FB --fb 1 --lim 0.1 --lim_mode box --N 640
  $R s_f_fX1_n640_r$r 0 $FB --fb 1 --N 640
  $R s_f_b03_n768_r$r 0 $FB --lim 0.3 --lim_mode box --N 768
done
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_r_ficus/s_f_*/pred.pt > $W/lbours/fem_r_ficus/period3.txt 2>&1
python exe/fem_r_table.py $W/lbours/fem_r_lego/s_l_* > $W/lbours/fem_r_lego/table3.md
python exe/fem_r_table.py $W/lbours/fem_r_ficus/s_f_* > $W/lbours/fem_r_ficus/table3.md
rclone copy $W/lbours/fem_r_lego r2:storage/result/anchorflow/lbours/fem_r_lego -q
rclone copy $W/lbours/fem_r_ficus r2:storage/result/anchorflow/lbours/fem_r_ficus -q
echo R3_DONE $(date +%T)
