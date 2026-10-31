#!/bin/bash
# 절 AK.14: 정답 MPM 민감도를 lego 21 전부로 (재현 2 회째 + 진단, 격자 141, 서브스텝 416) + 정답 FLIP 섞기 (미달 5 + 대조 2). 카드 1~6 비측정, 출력 /tmp/dkta/lbours/fem_ak_gt/<궤적>_<판> + R2.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; AF=$(pwd); C=$(git rev-parse --short HEAD)
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
GO=/tmp/dkta/lbours/fem_ak_gt; mkdir -p $GO; rm -f $GO/GTS2_DONE
gt(){ local t=$1 vk=$2 g=$3; shift 3; local o=$GO/${t}_$vk
  [ -f $o/err.json ] && return; mkdir -p $o; export WARP_CACHE_PATH=/tmp/dkta/warp_cache_$g; mkdir -p $WARP_CACHE_PATH
  if [ $t = L000 ]; then GA="--gt $W/lbdata/test_low/L000 --pt $W/lbpt/test_low/L000.pt"; else GA="--gt $W/lbdata/test/$t --pt /tmp/dkta/af_data/$t.pt"; fi
  echo "커밋 $C $*" > $o/cmd.txt
  (cd $W/i-physgaussian && CUDA_VISIBLE_DEVICES=$g python -u $AF/exe/gt_sens.py --out $o $GA --no_dump "$@" < /dev/null > $o/log.txt 2>&1)
  echo "$(date +%T) ${t}_$vk $(tail -1 $o/log.txt | cut -c1-120)"; rclone copy $o r2:storage/result/anchorflow/lbours/fem_ak_gt/${t}_$vk -q; }
JOBS=()
for t in 0001 0003 0016 0017 0018 0005 0013; do JOBS+=("$t flip95 --flip 0.95 --diag" "$t rep2 --diag"); done
for t in L000 0000 0002 0004 0006 0007 0008 0009 0010 0011 0012 0014 0019 0015; do JOBS+=("$t rep2 --diag"); done
for t in L000 0000 0002 0004 0006 0007 0008 0009 0010 0011 0012 0014 0019; do JOBS+=("$t rep" "$t ng141 --n_grid 141" "$t ns416 --nsub 416"); done
for g in 1 2 3 4 5 6; do
  ( k=0; for j in "${JOBS[@]}"; do if [ $((k % 6)) -eq $((g - 1)) ]; then set -- $j; t=$1; vk=$2; shift 2; gt $t $vk $g "$@"; fi; k=$((k + 1)); done ) &
done
wait
echo "[끝] $(date +%T)"; touch $GO/GTS2_DONE; rclone copy $GO r2:storage/result/anchorflow/lbours/fem_ak_gt -q
