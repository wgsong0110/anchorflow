#!/bin/bash
# 절 CM23.3 영상 (사용자 요청 11-07): 수박 기본 판 K 1 반복 상한별 3DGS 렌더 (일반 색) -- 정답 | 2x5 | 4x10 | 8x20 | 16x40.
#   wmk1conv 진단 판 (d_*) 에는 입자 덤프가 없어, 같은 명령 (절 CM23.3 진단 판 그대로) 에 --dump 만 붙여 다시 (v_*; 비결정이라 조금 다를 수 있음 -> 오차 곡선 대조는 로컬 합성 단계).
#   fem_ao_rt_sim 은 발산해도 그때까지 프레임을 pred.pt 에 쓴다 (--dump_part 없음, 필요 없음).
#   렌더 = exe/wm_render.py 색 바꿈 없이 (절 CM23 k1new 수박 렌더와 같은 경로: 유한 프레임 뒤 마지막 프레임으로 100 까지 채움 -> lb_roll_to_h5 -> wm_render).
#   정답 일반 색 영상은 R2 lbours/k1new/vid/gt_plain.mp4 (같은 렌더러·카메라) 재사용.
# 잡 안, 커밋 디렉토리에서; 출력 $1, R2 $2. 판마다 카드 하나 (시뮬 -> 렌더), 동시에.
O=$1; R=$2; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
WMD=/tmp/dkta/ag_data; mkdir -p $WMD
[ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
BASE="exe/fem_ao_rt_sim.py --data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh --fl_bar --fl_bst 1 --y0eb --lj_interp --fp_interp --tag wm --K 1"
DG="--nrec --qjd --jdiag --bar_cnt"
V=$O/vid; mkdir -p $V
log() { echo "[$(date '+%F %T')] $*" >> $O/wmk1vid.log; rclone copyto $O/wmk1vid.log $R/wmk1vid.log -q; }
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
# 정답 첫 프레임·meta (lb_roll_to_h5 --traj): 절 CM23 k1new 와 같은 자리
W=/home/dkta/work; CD=$(pwd -P)
WMM=$W/gfmodel/watermelon; [ -d $WMM/point_cloud ] || WMM=$W/gfmodel/model/watermelon; CFG=$W/wm2/gf/config.json; GTH=$W/wm2/gf/simulation_ply
GTD=/tmp/dkta/ag_data/gt_wm; [ -f $GTD/meta.json ] || GTD=/dev/shm/dkta/ag_data/gt_wm
if [ ! -f $GTD/meta.json ]; then
  g=$(claim_wait $$); mkdir -p /dev/shm/dkta/ag_data
  CUDA_VISIBLE_DEVICES=$g $PY -u exe/pg_h5_to_pt.py --h5_dir $GTH --config $CFG --out_pt /dev/shm/dkta/ag_data/wm_rend_tmp.pt --out_dir $GTD --out_dir_n 1 --allow_nonfinite > /dev/shm/dkta/ag_data/cv_wm_rend.log 2>&1
  rm -f /dev/shm/dkta/ag_data/wm_rend_tmp.pt $CL/$g; fi
log "정답 traj $GTD $(ls $GTD | head -3 | tr '\n' ' ')"
one() {   # one 이름 뉴턴 PCG
  local n=v_$1 nn=$2 pp=$3 me=$BASHPID g
  g=$(claim_wait $me); log "$n 카드 $g 시작"
  local c="$BASE --newton $nn --cg $pp $DG --dump"
  if [ ! -s $O/$n/pred.pt ] || [ ! -f $O/$n/summary.json ]; then
    rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/gpu_before.txt 2>&1
    CUDA_VISIBLE_DEVICES=$g timeout 5400 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
    log "$n 시뮬 rc=$? $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-220) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
    rclone copy $O/$n $R/$n -q --exclude pred.pt
  fi
  $PY - $O/$n/pred.pt $O/$n/pred_v.pt <<'PYX' > $V/pad_$1.log 2>&1
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False); P, F = d["pred"], d["F"]
n = 0
while n < P.shape[0] and bool(torch.isfinite(P[n]).all()) and bool(torch.isfinite(F[n]).all()): n += 1
L = 100
P, F = P[:n], F[:n]
if n < L:
    P = torch.cat([P, P[n - 1:n].expand(L - n, -1, -1)]); F = torch.cat([F, F[n - 1:n].expand(L - n, -1, -1, -1)])
torch.save({"pred": P.contiguous(), "F": F.contiguous()}, sys.argv[2]); print("덤프 프레임", d["pred"].shape[0], "유한 프레임", n, "->", P.shape[0])
PYX
  $PY -u exe/lb_roll_to_h5.py --dump $O/$n/pred_v.pt --traj $GTD --out $O/$n/h5_fem > $V/h5_$1.log 2>&1; rm -f $O/$n/pred_v.pt
  (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$g $PY -u $CD/exe/wm_render.py --model_path $WMM --config $CFG --frames 100 --sim $O/$n/h5_fem --out $V/$1.mp4 < /dev/null) > $V/r_$1.log 2>&1
  log "$n 렌더 $(tail -1 $V/pad_$1.log) $(tail -1 $V/h5_$1.log | cut -c1-150) $(tail -1 $V/r_$1.log | cut -c1-150)"
  rm -rf $O/$n/h5_fem; rm -f $CL/$g
  rclone copy $V $R/vid -q
  rclone copy $O/$n $R/$n -q                                          # 덤프 (pred.pt) 포함
}
for v in "N2P5 2 5" "N4P10 4 10" "N8P20 8 20" "N16P40 16 40"; do set -- $v; one $1 $2 $3 & sleep 3; done
wait
D=$(for n in v_N2P5 v_N4P10 v_N8P20 v_N16P40; do rclone check $O/$n $R/$n --one-way --size-only 2>&1 | grep -c ERROR; done | awk "{s+=\$1} END{print s}")
log "끝 R2 덤프 확인 차이 $D $(ls -la $V/*.mp4 2>&1 | awk '{print $5, $9}' | tr '\n' ' ')"
