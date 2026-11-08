#!/bin/bash
# 절 CM41 영상 (절 CM40 stab_wmshape_vid.sh 그대로, 목록만 exe/wmshape2_vid.txt): 판마다 손상 색 (logJp viridis −0.2~0.4) + 일반 3DGS 렌더 (같은 렌더러 exe/wm_render.py·카메라, 절 CM38.6 과 같은 절차).
# 잡 안, 커밋 디렉토리에서; 출력 $1, R2 $2. 목록 = exe/wmshape2_vid.txt 의 '<이름> run <플래그...>' (덤프 판으로 다시 돌림) 또는 '<이름> h5 <R2 h5_fem 경로>' (있는 덤프의 일반 렌더만).
O=$1; RO=$2
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
W=/home/dkta/work; CD=$(pwd -P)
eval "$(sed -n '/^WMD=/p;/^WMB=/p;/^base_of()/p;/^nw()/p' exe/stab_wmshape2.sh)"
log() { echo "[$(date '+%F %T')] $*" >> $O/vid.log; rclone copyto $O/vid.log $RO/vid.log -q; }
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
WMM=$W/gfmodel/watermelon; [ -d $WMM/point_cloud ] || WMM=$W/gfmodel/model/watermelon; CFG=$W/wm2/gf/config.json
GTM=/tmp/dkta/ag_data/gt_wm; [ -f $GTM/meta.json ] || GTM=/dev/shm/dkta/ag_data/gt_wm
one() { local n=$1 md=$2 x=$3 g=$4 Od=$O/vid/$1 c b V
  mkdir -p $Od/vj; V=$Od/vj
  if [ $md = run ]; then
    b="$(base_of M3)"; [ -n "$(nw "$x")" ] && b="${b/--newton 2/--newton $(nw "$x")}" && x="$(echo "$x" | sed 's/--newton [0-9]*//')"
    c="$b --erec --dump --dump_lj $x --out $Od"; echo "$PY -u $c" > $Od/cmd.txt
    CUDA_VISIBLE_DEVICES=$g timeout 7200 $PY -u $c < /dev/null > $Od/log.txt 2>&1
    log "[영상] $n rc=$? $(grep -ah '롤아웃\]' $Od/log.txt | tail -1 | cut -c1-200)"
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
    $PY -u exe/lb_roll_to_h5.py --dump $Od/pred_v.pt --traj $GTM --out $Od/h5_fem > $V/h5.log 2>&1; rm -f $Od/pred_v.pt
    $PY exe/wmdmg_stats.py --run $Od --sc_lj $Od/sc_lj > $V/sc.log 2>&1
    (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$g $PY -u $CD/exe/wm_render.py --model_path $WMM --config $CFG --sim $Od/h5_fem --scalar_dir $Od/sc_lj --vmin -0.2 --vmax 0.4 --cmap viridis --out $V/lj.mp4 < /dev/null > $V/r_lj.log 2>&1)
  else
    rclone copy $x $Od/h5_fem -q --transfers 16
  fi
  (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$g $PY -u $CD/exe/wm_render.py --model_path $WMM --config $CFG --sim $Od/h5_fem --out $V/plain.mp4 < /dev/null > $V/r_plain.log 2>&1)
  [ $md = run ] || rm -rf $Od/h5_fem
  log "[영상] $n lj $(ls -la $V/lj.mp4 2>&1 | awk '{print $5}') plain $(ls -la $V/plain.mp4 2>&1 | awk '{print $5}') $(tail -1 $V/r_plain.log | cut -c1-160)"
  rclone copy $Od $RO/vid/$n -q --transfers 8 --exclude pred.pt; }
while read -r n md x; do [ -z "$n" ] && continue; case $n in \#*) continue;; esac
  ( me=$BASHPID; g=$(claim_wait $me); one $n $md "$x" $g; rm -f $CL/$g ) & sleep 3
done < exe/wmshape2_vid.txt
wait
log "VID_END"
