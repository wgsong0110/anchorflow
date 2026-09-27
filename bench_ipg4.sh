#!/bin/bash
# i-PG 를 네 세팅으로 돌려 서로 비교한다 (제어점 이동이 큰 씬).
#   orig  : 격자 100^3, dt 배수 8   (원본에 가까운 세팅)
#   dtour : 격자 100^3, dt 프레임    (우리와 같은 시간 간격)
#   resour: 격자  32^3, dt 배수 8   (우리와 같은 해상도)
#   both  : 격자  32^3, dt 프레임    (둘 다 우리와 같게)
# 사용: bench_ipg4.sh <세팅> <조합> <시드> <GPU> [프레임]
W=${AF_WORK:-/home/dkta/work}
SET=$1; C=$2; SD=$3; GPU=$4; FR=${5:-20}
S=$(printf "%02d" $SD); SH_=${C%%_*}
export CUDA_VISIBLE_DEVICES=$GPU
export PATH=${AF_CONDA:-/home/dkta/.conda/envs/af/bin}:$PATH
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export AF_IPG_NEWTON=1 AF_H_R=0.15
# warp 커널 캐시를 세팅마다 분리한다. 같은 캐시를 여러 프로세스가 동시에 쓰면
# 컴파일 중인 모듈을 서로 읽어 'Failed to find forward kernel' 로 죽는다.
export WARP_CACHE_PATH=$W/.warpcache_$1
mkdir -p $WARP_CACHE_PATH
case $SET in
  orig)   NG=100; DTM=8;   ;;
  dtour)  NG=100; DTM=833; ;;
  resour) NG=32;  DTM=8;   ;;
  both)   NG=32;  DTM=833; ;;
  *) echo "세팅은 orig|dtour|resour|both"; exit 1;;
esac
CFG=$W/bench_cfg_4/${C}_${SET}.json
mkdir -p $W/bench_cfg_4 $W/bench_ipg4
python - "$W/bench_cfg/${C}.json" "$CFG" "$NG" "$FR" <<'PY'
import json, sys
src, dst, ng, fr = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
c = json.load(open(src)); c["n_grid"] = ng; c["frame_num"] = fr
json.dump(c, open(dst, "w"))
print(f"[설정] {dst} 격자 {ng}^3 프레임 {fr}")
PY
OUT=$W/bench_ipg4/${C}_s${S}_${SET}
rm -rf $OUT.h5dir && mkdir -p $OUT.h5dir
export AF_H_SCEN=$W/bench/scen_${C}_s${S}.npz AF_PGFILL_NPY=$W/pgfill_${SH_}.npy
T0=$SECONDS
(cd $W/i-physgaussian && python -u gs_simulation.py \
   --model_path $W/pgmodel/${SH_}_whitebg-trained --config $CFG \
   --output_path $OUT.h5dir --output_h5 --implicit --solver newton_gmres \
   --dt_multiplier $DTM --white_bg) > $W/bench_ipg4_${C}_${SET}.log 2>&1
echo "${SET} ${C} s${S} 격자${NG} dt x${DTM} $((SECONDS - T0)) s" >> $W/bench_ipg4_time.log
PYTHONPATH=$W/anchorflow/lib python -u $W/anchorflow/exe/h5_to_pt.py \
  --dir $OUT.h5dir --out $OUT.pt --cfg $CFG >> $W/bench_ipg4_${C}_${SET}.log 2>&1
echo "IPG4_DONE $SET" >> $W/bench_ipg4_progress.log
ls -la $OUT.pt 2>/dev/null
