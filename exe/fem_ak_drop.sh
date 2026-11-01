#!/bin/bash
# 절 AK.29 떨어지는 블록 단위 시험 (비측정): 자료 -> 정답 MPM -> 1/60·1/120·1/240 자료 -> 판 (상한 없는 판·기본 판·ES+γ) + 상한 없는 판 끄기 시험
#   (고착 몫 끔 = 미끄럼, 비관통 끔, 띠 0 끔 = 띠 0.5, 셀 변형 크기 = 프레임 1/120·1/240) -> 분석 -> R2.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; AF=$(pwd)
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
B=/tmp/dkta/ak_drop; mkdir -p $B
python exe/fem_ak_drop.py $B || exit 1
( export WARP_CACHE_PATH=/tmp/dkta/warp_cache_2; cd $W/i-physgaussian && CUDA_VISIBLE_DEVICES=2 python -u $AF/exe/fem_ak_drop_mpm.py $B/init.pt $B/mpm 80 > $B/mpm.log 2>&1 ); tail -2 $B/mpm.log
python exe/fem_ak_drop_mk.py $B || exit 1
export AK_KIND=F AK_EN=0 AK_W=1 AK_SFX=_drop
bash exe/fem_ak_sweep.sh v:nocs,cms,eshg:z:1:: drop1 > /dev/null 2>&1 &
bash exe/fem_ak_sweep.sh v:nocs:z:1:--vfl_mode@slip:_slip drop1 > /dev/null 2>&1 &
bash exe/fem_ak_sweep.sh v:nocs:z:1:--vfl_np@0:_np0 drop1 > /dev/null 2>&1 &
bash exe/fem_ak_sweep.sh v:nocs:g:1:: drop1 > /dev/null 2>&1 &
bash exe/fem_ak_sweep.sh v:nocs:z:1:: drop2,drop4 > /dev/null 2>&1 &
wait
R=/tmp/dkta/lbours/fem_ak_lego_drop
python exe/fem_ak_drop_ana.py $B/init.pt mpm:$B/mpm/x240.pt:0.0041666667 \
  nocs:$R/drop1_nocs_z_F1/roll_fem.pt:0.0166666667 cms:$R/drop1_cms_z_F1/roll_fem.pt:0.0166666667 eshg:$R/drop1_eshg_z_F1/roll_fem.pt:0.0166666667 \
  nocs_미끄럼:$R/drop1_nocs_z_slip_F1/roll_fem.pt:0.0166666667 nocs_비관통끔:$R/drop1_nocs_z_np0_F1/roll_fem.pt:0.0166666667 nocs_띠0.5:$R/drop1_nocs_g_F1/roll_fem.pt:0.0166666667 \
  nocs_프레임1/120:$R/drop2_nocs_z_F1/roll_fem.pt:0.0083333333 nocs_프레임1/240:$R/drop4_nocs_z_F1/roll_fem.pt:0.0041666667 2>&1 | tee $B/ana.txt
for d in $R/drop*_F1; do echo "$(basename $d) $(grep -a '롤아웃\]' $d/v_fem.log | cut -c1-150)"; done | tee -a $B/ana.txt
cat $B/info.json
rclone copy $B r2:storage/result/anchorflow/lbours/ak_drop -q
rclone copy $R r2:storage/result/anchorflow/lbours/fem_ak_lego_drop -q
echo "[끝] $(date +%T)"
