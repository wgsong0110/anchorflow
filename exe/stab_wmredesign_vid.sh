#!/bin/bash
# 절 CM38 영상: 통과 후보를 레고·wolf 결정적 + 덤프 (--dump --dump_F; 결과 불변) 로 다시 돌려 3DGS 렌더 (stab_common1_vid.sh 와 같은 절차), 수박은 덤프 판 (--erec --dump --dump_lj) 손상 색 렌더 (stab_wmen_vid.sh 와 같은 렌더러·카메라·범위).
# 잡 안, 커밋 디렉토리에서; 출력 $1, R2 $2. 후보 = exe/wmredesign_vid.txt 의 '<이름> <씬 L·W·M15·M3> <플래그...>' 줄.
O=$1; RO=$2
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
W=/home/dkta/work; CD=$(pwd -P)
source exe/fem_cm2_scn.sh
for t in L000 wolf; do cm_data $t; done
eval "$(sed -n '/^BA0=/p;/^FBB=/p;/^TL=/p;/^WMD=/p;/^WMB=/p;/^base_of()/,/^esac; }/p;/^nw()/p' exe/stab_wmredesign.sh)"
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
GTW=/tmp/dkta/ag_data/gt_wolf; [ -f $GTW/meta.json ] || GTW=/dev/shm/dkta/ag_data/gt_wolf
WMM=$W/gfmodel/watermelon; [ -d $WMM/point_cloud ] || WMM=$W/gfmodel/model/watermelon; CFG=$W/wm2/gf/config.json
GTM=/tmp/dkta/ag_data/gt_wm; [ -f $GTM/meta.json ] || GTM=/dev/shm/dkta/ag_data/gt_wm
one() { local n=$1 sc=$2 x=$3 g=$4 Od=$O/vid/$1 c b GTD SH CF V
  mkdir -p $Od/vj; V=$Od/vj
  b="$(base_of $sc)"; [ -n "$(nw "$x")" ] && b="${b/--newton 2/--newton $(nw "$x")}" && x="$(echo "$x" | sed 's/--newton [0-9]*//')"
  case $sc in
  L|W)
    if [ $sc = L ]; then GTD=$W/lbdata/test_low/L000; SH=lego; CF=$W/lbdata/config.json; else GTD=$GTW; SH=wolf; CF=$GTW/config.json; fi
    c="$b --det --fhash --dump --dump_F $x --out $Od"; echo "$PY -u $c" > $Od/cmd.txt
    CUDA_VISIBLE_DEVICES=$g timeout 3600 $PY -u $c < /dev/null > $Od/log.txt 2>&1
    $PY -u exe/lb_roll_to_h5.py --dump $Od/pred.pt --traj $GTD --out $Od/h5 > $Od/h5.log 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY -u exe/lb_render.py --sim $Od/h5 --config $CF --shape $SH --out $V/fem_$SH.mp4 --frame_label 0 --caption " " < /dev/null > $V/r.log 2>&1
    rm -rf $Od/h5
    log "[렌더] $n $(grep -ah '롤아웃\]' $Od/log.txt | tail -1 | cut -c1-160) $(ls -la $V/fem_$SH.mp4 2>&1 | awk '{print $5}') $(tail -1 $V/r.log | cut -c1-160)";;
  M*)
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
    rm -rf $Od/h5_fem $Od/sc_lj
    log "[영상] $n $(tail -1 $V/pad.log) lj $(ls -la $V/lj.mp4 2>&1 | awk '{print $5}') $(tail -1 $V/r_lj.log | cut -c1-200)";;
  esac
  rclone copy $Od $RO/vid/$n -q --transfers 8; }
while read -r n sc x; do [ -z "$n" ] && continue; case $n in \#*) continue;; esac
  ( me=$BASHPID; g=$(claim_wait $me); one $n $sc "$x" $g; rm -f $CL/$g ) & sleep 3
done < exe/wmredesign_vid.txt
wait
log "VID_END"
