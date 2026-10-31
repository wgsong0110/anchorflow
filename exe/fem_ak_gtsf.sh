#!/bin/bash
# 절 AK.15: ficus 정답 (PG gs_simulation 형식) 민감도 -- 재현 2 회, 서브스텝 200 (dt 2 배), 격자 71 (dx 0.028), 격자 35 (dx 0.057 = 우리 node_h). 카드 AK_G (기본 1~4), 출력 /tmp/dkta/lbours/fem_ak_gt/ficus_<판> + R2.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; AF=$(pwd); C=$(git rev-parse --short HEAD)
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
GO=/tmp/dkta/lbours/fem_ak_gt; mkdir -p $GO
FA="--gt $W/lbours/ficus_v0/gt --config $W/lbours/ficus_v0/gt/config.json --pt $W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --frames 125 --pg_cfg --no_dump"
gt(){ local vk=$1 g=$2; shift 2; local o=$GO/ficus_$vk
  [ -f $o/err.json ] && return; mkdir -p $o; export WARP_CACHE_PATH=/tmp/dkta/warp_cache_$g; mkdir -p $WARP_CACHE_PATH
  echo "커밋 $C $*" > $o/cmd.txt
  (cd $W/i-physgaussian && CUDA_VISIBLE_DEVICES=$g python -u $AF/exe/gt_sens.py --out $o $FA "$@" < /dev/null > $o/log.txt 2>&1)
  echo "$(date +%T) ficus_$vk $(tail -1 $o/log.txt | cut -c1-140)"; rclone copy $o r2:storage/result/anchorflow/lbours/fem_ak_gt/ficus_$vk -q; }
# warp 가 경계조건 지역 커널을 카드별 캐시에서 동시에 처음 만들면 못 찾는다 (실측) -- 한 카드에서 차례로
( gt ng35 ${AK_G:-4} --n_grid 35; gt rep ${AK_G:-4}; gt rep2 ${AK_G:-4}; gt ns200 ${AK_G:-4} --nsub 200; gt ng71 ${AK_G:-4} --n_grid 71 ) &
wait
touch $GO/GTSF_DONE; echo "[끝] $(date +%T)"
