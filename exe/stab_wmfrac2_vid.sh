#!/bin/bash
# 절 CM32 영상: 기준 판 (REF) 과 가장 좋은 판 (BEST) 을 덤프 판 (--erec --dump --dump_lj) 으로 다시 돌려 손상 색 (logJp viridis −0.2~0.4, 절 CM28.4·CM30.3 과 같은 렌더러·카메라·범위) 3DGS 렌더.
#   정답 칸 = R2 lbours/fem_cm/wmJd/gt_lj.mp4 (로컬 합성). 잡 안, 커밋 디렉토리에서: bash exe/stab_wmfrac2_vid.sh <출력> <R2> <가장 좋은 판> <기준 판>
O=$1; R=$2
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
log() { echo "[$(date '+%F %T')] $*" >> $O/wmfrac2.log; rclone copyto $O/wmfrac2.log $R/wmfrac2.log -q; }
CL=/dev/shm/dkta/anerr_claim; mkdir -p $CL
NG=$(nvidia-smi -L | wc -l)
claim() { local me=$1 g f p
  for g in $(seq 0 $((NG - 1))); do
    f=$CL/$g
    if [ -f $f ]; then p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && continue; rm -f $f; fi
    [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" = 0 ] || continue
    ( set -o noclobber; echo $me > $f ) 2>/dev/null && { echo $g; return 0; }
  done; return 1; }
claim_wait() { local g; while :; do g=$(claim $1) && { echo $g; return; }; sleep 30; done; }
BEST=$3; REF=$4
echo "$REF $BEST" > $O/video_runs.txt; rclone copy $O $R -q --include video_runs.txt
log "[영상] 기준 판 $REF, 가장 좋은 판 $BEST"
W=/home/dkta/work; CD=$(pwd -P)
WMM=$W/gfmodel/watermelon; [ -d $WMM/point_cloud ] || WMM=$W/gfmodel/model/watermelon; CFG=$W/wm2/gf/config.json
GTD=/tmp/dkta/ag_data/gt_wm; [ -f $GTD/meta.json ] || GTD=/dev/shm/dkta/ag_data/gt_wm
eval "$(sed -n '/^BASE=/p;/^WMD=/p;/^atom()/,/esac; }/p;/^cmd_of()/,/echo "\$c"; }/p' exe/stab_wmfrac2.sh)"
one() { local v=$1 n=vid_$1 me=$BASHPID g Od V
  Od=$O/$n; V=$Od/vid
  [ -s $V/lj.mp4 ] && return
  g=$(claim_wait $me); log "[영상] $n 카드 $g"
  if [ ! -f $Od/summary.json ]; then
    c="$(cmd_of $v) --erec --dump --dump_lj"; rm -rf $Od; mkdir -p $Od; echo "$PY -u $c --out $Od" > $Od/cmd.txt
    CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c --out $Od < /dev/null > $Od/log.txt 2>&1
    log "[영상] $n rc=$? $(grep -ah '롤아웃\]' $Od/log.txt | tail -1 | cut -c1-200)"
  fi
  mkdir -p $V
  if [ ! -f $GTD/meta.json ]; then GTD=/dev/shm/dkta/ag_data/gt_wm_$v; mkdir -p $GTD
    CUDA_VISIBLE_DEVICES=$g $PY -u exe/pg_h5_to_pt.py --h5_dir $W/wm2/gf/simulation_ply --config $CFG --out_pt $GTD.tmp.pt --out_dir $GTD --out_dir_n 1 --allow_nonfinite > $V/cv.log 2>&1; rm -f $GTD.tmp.pt; fi
  $PY - $Od/pred.pt $Od/pred_v.pt <<'PYX' > $V/pad.log 2>&1
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False); P, F = d["pred"], d["F"]
n = 0
while n < P.shape[0] and bool(torch.isfinite(P[n]).all()) and bool(torch.isfinite(F[n]).all()): n += 1
L = 100
P, F = P[:n], F[:n]
if n < L:
    P = torch.cat([P, P[n - 1:n].expand(L - n, -1, -1)]); F = torch.cat([F, F[n - 1:n].expand(L - n, -1, -1, -1)])
torch.save({"pred": P.contiguous(), "F": F.contiguous()}, sys.argv[2]); print("유한 프레임", n, "->", P.shape[0])
PYX
  $PY -u exe/lb_roll_to_h5.py --dump $Od/pred_v.pt --traj $GTD --out $Od/h5_fem > $V/h5.log 2>&1; rm -f $Od/pred_v.pt
  $PY exe/wmdmg_stats.py --run $Od --sc_lj $Od/sc_lj > $V/sc.log 2>&1
  (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$g $PY -u $CD/exe/wm_render.py --model_path $WMM --config $CFG --sim $Od/h5_fem --scalar_dir $Od/sc_lj --vmin -0.2 --vmax 0.4 --cmap viridis --out $V/lj.mp4 < /dev/null > $V/r_lj.log 2>&1)
  rm -f $CL/$g
  rm -rf $Od/h5_fem $Od/sc_lj
  log "[영상] $n $(tail -1 $V/pad.log) lj $(ls -la $V/lj.mp4 2>&1 | awk '{print $5}') $(tail -1 $V/r_lj.log | cut -c1-200)"
  rclone copy $Od $R/$n -q --transfers 8 --s3-upload-concurrency 16 --s3-chunk-size 64M; }
one $REF & sleep 5
one $BEST &
wait
log "[영상] 끝"
