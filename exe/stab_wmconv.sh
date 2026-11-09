#!/bin/bash
# 절 CM90 묶음 wmconv: 수렴 판 (jC) 2 판 + H9 그대로 (jA_3) 1 판을 덤프 (exe/wmconv_wrap.py: 위치·F·logJp, 결과 불변) 와 함께 다시 돌리고,
# 판마다 같은 카드에서 손상 통계 (exe/wmdmg_stats.py, exe/wmconv_stat.py) + 영상 칸 (손상 색 logJp viridis −0.2~0.4 + 일반 3DGS, exe/wm_render.py; exe/stab_wmshape4_vid.sh 와 같은 절차).
# 잡 안, 코드 디렉토리에서. 노드 / ·/tmp 안 씀 (코드·자료·TMPDIR·결과·h5 /dev/shm/dkta). 카드는 공용 표식 디렉토리로 계산 프로세스 0 인 카드만.
# 사용: bash exe/stab_wmconv.sh <출력> <R2>   (판 목록 exe/wmconv_spec.txt)
O=$1; RO=$2; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR
W=/home/dkta/work; CD=$(pwd -P)
AG=/dev/shm/dkta/ag_data; mkdir -p $AG
[ -s $AG/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $AG/wm.pt.$$ -q && mv $AG/wm.pt.$$ $AG/wm.pt; }
GTM=$AG/gt_wm
JP=$AG/gt_jp; mkdir -p $JP
[ $(ls $JP | wc -l) -ge 101 ] || rclone copy r2:storage/result/anchorflow/lbours/fem_cm/wmJd/gt_jp $JP -q --transfers 16
WMM=$W/gfmodel/watermelon; [ -d $WMM/point_cloud ] || WMM=$W/gfmodel/model/watermelon; CFG=$W/wm2/gf/config.json
log() { echo "[$(date '+%F %T')] $*" >> $O/wmconv.log; rclone copyto $O/wmconv.log $RO/wmconv.log -q; }
shm_use() { df --output=pcent /dev/shm | tail -1 | tr -dc '0-9'; }
log "시작 $(hostname) 코드 $CD; wm.pt $(ls -la $AG/wm.pt | awk '{print $5}') gt_wm $(ls $GTM/meta.json 2>&1) gt_jp $(ls $JP | wc -l) 모델 $WMM; /dev/shm $(shm_use) %"
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
one() { local n=$1 c=$2 g=$3 Od=$O/$1 V
  [ -f $Od/summary.json ] && [ -f $Od/vj/plain.mp4 ] && return
  rm -rf $Od; mkdir -p $Od/vj; V=$Od/vj
  c="${c//@OUT@/$Od}"; echo "$c" > $Od/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $Od/apps_at_start.txt 2>&1; echo "카드 $g" >> $Od/apps_at_start.txt; nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader >> $Od/apps_at_start.txt; cat /proc/loadavg >> $Od/apps_at_start.txt
  while [ "$(shm_use)" -ge 65 ]; do log "$n /dev/shm $(shm_use) % -- 기다림"; sleep 120; done
  log "$n 카드 $g 시작"
  CUDA_VISIBLE_DEVICES=$g timeout 5400 $c < /dev/null > $Od/log.txt 2>&1
  log "$n rc=$? $(grep -ah '롤아웃\]' $Od/log.txt | tail -1 | cut -c1-200) $(grep -ah 'wmconv\] 덤프 [0-9]' $Od/log.txt | tail -1) $(grep -a 'rror' $Od/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $Od $RO/$n -q --transfers 8 --exclude pred.pt --exclude lj.pt --exclude "vj/**"
  [ -f $Od/pred.pt ] && [ -f $Od/lj.pt ] || { log "$n 덤프 없음"; return; }
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
  $PY exe/wmconv_stat.py --run $Od --jp $JP > $Od/wmconv_stat.log 2>&1
  (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$g $PY -u $CD/exe/wm_render.py --model_path $WMM --config $CFG --sim $Od/h5_fem --scalar_dir $Od/sc_lj --vmin -0.2 --vmax 0.4 --cmap viridis --out $V/lj.mp4 < /dev/null > $V/r_lj.log 2>&1)
  (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$g $PY -u $CD/exe/wm_render.py --model_path $WMM --config $CFG --sim $Od/h5_fem --out $V/plain.mp4 < /dev/null > $V/r_plain.log 2>&1)
  log "$n 영상 lj $(ls -la $V/lj.mp4 2>&1 | awk '{print $5}') plain $(ls -la $V/plain.mp4 2>&1 | awk '{print $5}') $(tail -1 $V/pad.log) $(tail -1 $Od/wmconv_stat.log | cut -c1-400)"
  rm -rf $Od/h5_fem $Od/sc_lj
  rclone copy $Od $RO/$n -q --transfers 8 --exclude pred.pt
  rclone copyto $Od/pred.pt $RO/$n/pred.pt -q
  [ "$(rclone size $RO/$n/pred.pt --json 2>/dev/null | grep -o '"bytes":[0-9]*' | cut -d: -f2)" = "$(stat -c %s $Od/pred.pt)" ] && rm -f $Od/pred.pt && log "$n pred.pt R2 확인 뒤 로컬 사본 지움"; }
while read -r n c; do [ -z "$n" ] && continue; case $n in \#*) continue;; esac
  ( me=$BASHPID; g=$(claim_wait $me); one $n "$c" $g; rm -f $CL/$g ) & sleep 3
done < exe/wmconv_spec.txt
wait
log "끝"
