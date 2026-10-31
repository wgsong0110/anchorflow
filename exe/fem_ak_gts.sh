#!/bin/bash
# 절 AK.3 정답 MPM 민감도 (홀드아웃): 정답 솔버 (PG 공식 warp, exe/gt_sens.py) 를 조건 하나씩 바꿔 다시 굴려 정답과의 차 -- 이 궤적들이 이산화 세부에 얼마나 갈리는지 (도달 가능한 오차 바닥).
# 궤적 x 판 (재현 / 서브스텝 절반 416 / 격자 141 / 바닥 높이 +반 칸 0.49) 을 카드 1~6 에 나눠 (비측정). 출력 /tmp/dkta/lbours/fem_ak_gt/<궤적>_<판> + R2.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; AF=$(pwd); C=$(git rev-parse --short HEAD)
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
GO=/tmp/dkta/lbours/fem_ak_gt; mkdir -p $GO; rm -f $GO/GTS_DONE
TS=${AK_TS:-"0001 0003 0015 0016 0017 0018 0005 0013"}
gt(){ local t=$1 vk=$2 g=$3; shift 3; local o=$GO/${t}_$vk
  [ -f $o/err.json ] && return; mkdir -p $o; export WARP_CACHE_PATH=/tmp/dkta/warp_cache_$g; mkdir -p $WARP_CACHE_PATH   # /home 이 가득 -- warp 커널 캐시를 노드 /tmp 에 (카드별, 동시 컴파일 충돌 방지) echo "커밋 $C $*" > $o/cmd.txt
  (cd $W/i-physgaussian && CUDA_VISIBLE_DEVICES=$g python -u $AF/exe/gt_sens.py --out $o --gt $W/lbdata/test/$t --pt /tmp/dkta/af_data/$t.pt --no_dump "$@" < /dev/null > $o/log.txt 2>&1)
  echo "$(date +%T) ${t}_$vk $(tail -1 $o/log.txt | cut -c1-160)"; rclone copy $o r2:storage/result/anchorflow/lbours/fem_ak_gt/${t}_$vk -q; }
JOBS=()
for t in $TS; do JOBS+=("$t rep" "$t ns416 --nsub 416" "$t ng141 --n_grid 141" "$t fz049 --floor_z 0.49" "$t ng71 --n_grid 71" "$t pic --rpic -1"); done
i=0; for g in 1 2 3 4 5 6; do
  ( k=0; for j in "${JOBS[@]}"; do if [ $((k % 6)) -eq $((g - 1)) ]; then set -- $j; t=$1; vk=$2; shift 2; gt $t $vk $g "$@"; fi; k=$((k + 1)); done ) &
done
wait
echo "[정답 민감도 끝] $(date +%T)"; touch $GO/GTS_DONE; rclone copy $GO r2:storage/result/anchorflow/lbours/fem_ak_gt -q
