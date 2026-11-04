#!/bin/bash
# 절 AQ 렌더 (잡 안): 재질 씬 (bread·수박) 판 덤프 pred.pt (pred·F) -> lb_roll_to_h5 -> 3DGS 알파블렌딩 (절 AG 렌더와 같은 렌더러: bread lb_render, 수박 GF wm_render).
# 정답 영상은 절 AG 의 같은 렌더러 판 (R2 lbours/fem_ag_<씬>/gt) 을 그대로 쓴다 (정답 | 우리 합성은 로컬에서).
# 사용: AO_CARD=<카드> bash exe/fem_aq_render.sh <묶음> <씬 bread|wm> <판> ...   출력 R2 lbours/fem_ao/<묶음>/<판>/vj/fem_<형상>.mp4
cd "$(dirname "$0")/.." || exit 1
WT=$(pwd)
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8 PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH
export CUPY_CACHE_DIR=/tmp/dkta/cache/cupy XDG_CACHE_HOME=/tmp/dkta/cache TORCH_EXTENSIONS_DIR=/tmp/dkta/cache/torch_ext
source exe/fem_ag_scene.sh
RR=r2:storage/result/anchorflow/lbours/fem_ao
NM=$1; S=$2; shift 2; G=${AO_CARD:?카드}
ag_scene $S
WMM=$W/gfmodel/watermelon; [ -d $WMM/point_cloud ] || WMM=$W/gfmodel/model/watermelon
echo "[렌더 카드 $G] $S $(date +%T)"
for V in "$@"; do
  O=/tmp/dkta/lbours/fem_ao/$NM/$V
  [ -f $O/pred.pt ] || { mkdir -p $O; rclone copyto $RR/$NM/$V/pred.pt $O/pred.pt -q; }
  [ -f $O/pred.pt ] || { echo "[덤프 없음] $V"; continue; }
  mkdir -p $O/vj
  [ -f $O/h5_fem/meta.json ] || $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $AGD/gt_$S --out $O/h5_fem > $O/h5_fem.log 2>&1
  tail -1 $O/h5_fem.log
  if [ $S = wm ]; then
    [ -f $O/vj/fem_$SHAPE.mp4 ] || (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$G $PY -u $WT/exe/wm_render.py --model_path $WMM --config $CFG --sim $O/h5_fem --out $O/vj/fem_$SHAPE.mp4 < /dev/null > $O/vj/r_fem.log 2>&1)
  else
    [ -f $O/vj/fem_$SHAPE.mp4 ] || CUDA_VISIBLE_DEVICES=$G $PY -u exe/lb_render.py --sim $O/h5_fem --config $AGD/gt_$S/config.json --shape $SHAPE --out $O/vj/fem_$SHAPE.mp4 \
       --frame_label 0 --caption " " < /dev/null > $O/vj/r_fem.log 2>&1
  fi
  [ -f $O/vj/fem_$SHAPE.mp4 ] && echo "[렌더] $V" || tail -5 $O/vj/r_fem.log
  rclone copy $O/vj $RR/$NM/$V/vj -q
  rclone copy $O/h5_fem $RR/$NM/$V/h5_fem -q
done
echo "[렌더 끝] $(date +%T)"
