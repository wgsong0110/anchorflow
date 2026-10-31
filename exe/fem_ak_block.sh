#!/bin/bash
# 절 AK.24 블록 단위 시험: 자료 만들기 -> 정답 MPM (카드 2) / 우리 판 변형 (카드 3~6, fem_ab_run 비측정, f32 덤프) -> 분석.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; AF=$(pwd)
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
B=/tmp/dkta/ak_block; mkdir -p $B
[ -f $B/shear.pt ] || python exe/fem_ak_block.py $B 1.0 0.03 0.0125
for k in stretch shear; do
  ( export WARP_CACHE_PATH=/tmp/dkta/warp_cache_2; cd $W/i-physgaussian && [ -f $B/mpm_$k/x.pt ] || CUDA_VISIBLE_DEVICES=2 python -u $AF/exe/fem_ak_block_mpm.py $B/$k.pt $B/mpm_$k 80 > $B/mpm_$k.log 2>&1 ) &
done
BASE="--frames 80 --node_h 0.028284 --N 267 --floor none --mode lag --opt --mass tet --rstat --cfuse --cfuse_mode 2 --lim_sr 1e30 --ak_pq --ak_pq4 --ah_fast --ak_prep --ak_fuse --ak_sfcr --dump_f32"
declare -A VS=( [cms]="--sp vcap --sp_vcap 0.5 --ak_capmu --ak_fstep" [nocap]="--sp vcap --sp_vcap 1e9 --ak_fstep" [mcfl]="--sp vcap --sp_vcap 0.5 --ak_capmu --ah_mcfl 1" [b1]="--sp vcap --sp_vcap 0.5 --ak_capmu --ak_fstep --fupd navg --fupd_b 1" )
i=0
for k in stretch shear; do for vk in cms nocap mcfl b1; do
  G=$((3 + i % 4)); i=$((i + 1))
  EX_ROOT=fem_ak_block AB_OUT=/tmp/dkta/lbours bash exe/fem_ab_run.sh ${k}_$vk $G --data $B/$k.pt --tag ${k}_$vk $BASE ${VS[$vk]} < /dev/null > /dev/null 2>&1 &
done; done
wait
for k in stretch shear; do
  python exe/fem_ak_block_ana.py $B/$k.pt mpm:$B/mpm_$k/x.pt $(for vk in cms nocap mcfl b1; do echo "$vk:/tmp/dkta/lbours/fem_ak_block/${k}_$vk/roll_fem.pt"; done)
done | tee $B/ana.txt
rclone copy $B/ana.txt r2:storage/result/anchorflow/lbours/fem_ak_block -q
echo "[끝] $(date +%T)"
