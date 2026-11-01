#!/bin/bash
# 절 AO 렌더 (잡 안): 판 덤프 pred.pt (pred·F) -> lb_roll_to_h5 -> lb_render (3DGS 알파블렌딩, 절 AG 렌더와 같은 렌더러·config·형상) + 정답 같은 렌더러 (R2 에 있으면 그것).
# 사용: AO_CARD=<카드> bash exe/fem_ao_render.sh <묶음> <판> ...   출력 <판>/vj/fem_wolf.mp4 -> R2 lbours/fem_ao/<묶음>/<판>/vj
cd "$(dirname "$0")/.." || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8 PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH
export CUPY_CACHE_DIR=/tmp/dkta/cache/cupy XDG_CACHE_HOME=/tmp/dkta/cache TORCH_EXTENSIONS_DIR=/tmp/dkta/cache/torch_ext
W=/home/dkta/work; AGD=/tmp/dkta/ag_data; RR=r2:storage/result/anchorflow/lbours/fem_ao
NM=$1; shift; G=${AO_CARD:-auto}
if [ $G = auto ]; then                                                                 # 4·5·6 중 빈 카드를 기다려 잡음
  for w in $(seq 240); do
    nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader > /tmp/dkta/ao_rsmi.txt; nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader > /tmp/dkta/ao_rapps.txt
    G=$(awk -F', ' 'FILENAME ~ /rapps/ {b[$1]=1; next} {m=$3; sub(/ MiB/, "", m); if (!($2 in b) && ($1 == 4 || $1 == 5 || $1 == 6) && m + 0 < 2000) {print $1; exit}}' /tmp/dkta/ao_rapps.txt /tmp/dkta/ao_rsmi.txt)
    [ -n "$G" ] && break; sleep 30
  done
fi
echo "[렌더 카드 $G] $(date +%T)"
GTD=$AGD/gt_wolf
if [ ! -f $GTD/meta.json ]; then
  $PY -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /tmp/dkta/ag_data/wolf_rend_tmp.pt --out_dir $GTD > /tmp/dkta/ag_data/cv_wolf_rend.log 2>&1
  tail -2 /tmp/dkta/ag_data/cv_wolf_rend.log
fi
for V in "$@"; do
  O=/tmp/dkta/lbours/fem_ao/$NM/$V
  [ -f $O/pred.pt ] || { mkdir -p $O; rclone copyto $RR/$NM/$V/pred.pt $O/pred.pt -q; }                   # 노드에서 지웠으면 R2 에서
  [ -f $O/pred.pt ] || { echo "[덤프 없음] $V"; continue; }
  mkdir -p $O/vj
  [ -f $O/h5_fem/meta.json ] || $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $GTD --out $O/h5_fem > $O/h5_fem.log 2>&1
  tail -1 $O/h5_fem.log
  [ -f $O/vj/fem_wolf.mp4 ] || CUDA_VISIBLE_DEVICES=$G $PY -u exe/lb_render.py --sim $O/h5_fem --config $GTD/config.json --shape wolf --out $O/vj/fem_wolf.mp4 \
     --frame_label 0 --caption " " < /dev/null > $O/vj/r_fem.log 2>&1
  [ -f $O/vj/fem_wolf.mp4 ] && echo "[렌더] $V" || tail -5 $O/vj/r_fem.log
  rclone copy $O/vj $RR/$NM/$V/vj -q
done
if ! rclone lsf r2:storage/result/anchorflow/lbours/fem_ag_wolf/gt/gt_wolf.mp4 -q > /dev/null 2>&1; then
  mkdir -p /tmp/dkta/lbours/fem_ao/gt
  CUDA_VISIBLE_DEVICES=$G $PY -u exe/lb_render.py --sim $GTD --config $GTD/config.json --shape wolf --out /tmp/dkta/lbours/fem_ao/gt/gt_wolf.mp4 --frame_label 0 --caption " " < /dev/null > /tmp/dkta/lbours/fem_ao/gt/r_gt.log 2>&1
  rclone copy /tmp/dkta/lbours/fem_ao/gt $RR/gt -q
fi
echo "[렌더 끝] $(date +%T)"
