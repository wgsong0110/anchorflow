#!/bin/bash
# 절 CM21 묶음 wmbar (사용자 승인 11-07 정의 변경): 수박 장벽 바닥 + logJp 가우스 점 보간 (+ F 보간). 잡 안, 커밋 디렉토리에서; 출력 $1, R2 $2.
# 판 (모두 절 CM13 s_a 명령 = exe/fem_cm9_scn.sh wm + --gq_mass --elx snh, 셀 1.5, 비결정이라 각 2 회 a·b):
#   s0 = 지금 SNH 판 (GF 바닥 z 0.02 미끄럼 --floor_slip) -- 같은 묶음 대조
#   b1 = (1) 바닥만 장벽 바닥: --floor_slip 대신 --fl_bar --fl_bst 1 --y0eb (레고 (B) 정의, 같은 평면)
#   b2 = (2) (1) + --lj_interp (입자 logJp = 프레임 끝 가우스 점 logJp 보간)
#   b3 = (3) (2) + --fp_interp (입자 F 도 보간; 절 CM20 정의·플래그 그대로 -- 같은 가중으로 logJp 까지 교체)
#   판정 판 진단 (결과 불변): --gq_chk --dump --nan_step --jdiag --dump_lj (+ 장벽 판 --bar_cnt)
#   단독 시간 z_<판>: --gq_mass --elx snh + 판 플래그만 (진단·덤프 없음), 카드 하나 잡고 차례로, 그 카드·노드 계산 프로세스 수 10 초마다
# 카드: 공용 잡기 규약 (/dev/shm/dkta/anerr_claim/<카드>, 그 카드 계산 프로세스 0 일 때만), 카드 하나에 판 하나. 판이 끝나면 바로 R2 (pred.pt 포함).
O=$1; R=$2; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
WMD=/tmp/dkta/ag_data; mkdir -p $WMD
[ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
WMC="--data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut --tolN 1e-2 --tolP 1e-1 --K 3 --newton 2 --cg 5 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh"
DG="--gq_chk --dump --nan_step --jdiag --dump_lj"
BAR="--fl_bar --fl_bst 1 --y0eb"
add_of() { case $1 in s0) echo "--floor_slip";; b1) echo "$BAR";; b2) echo "$BAR --lj_interp";; b3) echo "$BAR --lj_interp --fp_interp";; esac; }
dg_of() { case $1 in s0) echo "$DG";; *) echo "$DG --bar_cnt";; esac; }
up() { rclone copy $O/$1 $R/$1 -q --transfers 8; rclone copy $O $R -q --include "wmbar.log" --include "wmbar_table.md"
  [ -s $O/$1/pred.pt ] && rclone check $O/$1 $R/$1 --one-way --size-only -q --include pred.pt && rm -f $O/$1/pred.pt; }   # /dev/shm 자리: 덤프 (6.6 GB) 는 R2 확인 뒤 지움 (렌더 때 다시 받음)
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
log() { echo "[$(date '+%F %T')] $*" >> $O/wmbar.log; }
run() {    # run 판 이름 카드 [frames]
  local r=$1 n=$2 g=$3 nf=${4:-} c
  [ -f $O/$n/summary.json ] && return
  c="$WMC $(add_of $r) $(dg_of $r)"; [ -n "$nf" ] && c=$(echo "$c" | sed "s/--frames 100/--frames $nf/")
  rm -rf $O/$n; mkdir -p $O/$n
  echo "$PY -u exe/fem_ao_rt_sim.py $c --tag wm --out $O/$n" > $O/$n/cmd.txt
  echo "카드 $g $(git log --oneline -1 | cut -c1-12) $(date '+%F %T')" > $O/$n/run_info.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1
  log "$n 카드 $g 시작"
  CUDA_VISIBLE_DEVICES=$g CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_wmbar_$g timeout 7200 $PY -u exe/fem_ao_rt_sim.py $c --tag wm --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  [ -s $O/$n/lj.pt ] && [ -s $O/$n/pred.pt ] && $PY exe/wmdmg_stats.py --run $O/$n > $O/$n/dmg_stats.log 2>&1
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(tail -1 $O/$n/dmg_stats.log 2>/dev/null | cut -c1-160) $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n
}
log "시작 $(hostname) $(git log --oneline -1 | cut -c1-80)"
# ---- 연기: (1)·(3) 3 프레임 (새 커널 컴파일·실행) -- 실패면 멈춤
if [ ! -f $O/smoke_ok ]; then
  ( g=$(claim_wait $BASHPID); run b1 smoke_b1 $g 3; rm -f $CL/$g ) & sleep 5
  ( g=$(claim_wait $BASHPID); run b3 smoke_b3 $g 3; rm -f $CL/$g ) & wait
  if [ -f $O/smoke_b1/summary.json ] && [ -f $O/smoke_b3/summary.json ]; then touch $O/smoke_ok; log "[연기] 통과 $(grep -o '"bc": {[^}]*}' $O/smoke_b1/fem.jsonl | tail -1) $(grep -o '"y0k": \[[^]]*\]' $O/smoke_b1/fem.jsonl | tail -1)"
  else log "[연기 실패] $(tail -4 $O/smoke_b1/log.txt $O/smoke_b3/log.txt | tr '\n' ' ' | cut -c1-1500)"; echo "연기 실패 $(date '+%F %T')" > $O/wmbar.FAIL; exit 1; fi
fi
# ---- 판정 8 판: 공용 대기열, 줄마다 카드를 잡고 판 하나
Q=$O/queue.txt; [ -f $Q ] || { for r in b3 b2 b1 s0; do echo "$r ${r}_a"; done; for r in b3 b2 b1 s0; do echo "$r ${r}_b"; done; } > $Q
echo 0 > $O/qpos
lane() { local me=$BASHPID i ln g
  while :; do
    i=$(flock $O/qpos.lock bash -c "i=\$(cat $O/qpos); echo \$((i + 1)) > $O/qpos; echo \$i")
    ln=$(sed -n "$((i + 1))p" $Q); [ -n "$ln" ] || break
    set -- $ln; [ -f $O/$2/summary.json ] && continue
    g=$(claim_wait $me); run $1 $2 $g; rm -f $CL/$g
  done; }
for k in 1 2 3 4 5 6 7 8; do lane & sleep 3; done
wait
log "판정 판 끝"
$PY exe/wmbar_tab.py $O > $O/wmbar_table.md 2> $O/wmbar_table.err; up smoke_b1
# ---- 단독 시간: 카드 하나 잡고 차례로 (비결정, 진단·덤프 없음)
zrun() { local r=$1 n=z_$1 g=$2 c
  [ -f $O/$n/summary.json ] && return
  c="$WMC $(add_of $r)"; rm -rf $O/$n; mkdir -p $O/$n
  echo "$PY -u exe/fem_ao_rt_sim.py $c --tag wm --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/gpu_before.txt 2>&1
  ( while sleep 10; do echo "$(date '+%T') 카드 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .) 노드 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" >> $O/$n/gpu_procs.txt; done ) & local wp=$!
  CUDA_VISIBLE_DEVICES=$g CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_wmbar_$g timeout 7200 $PY -u exe/fem_ao_rt_sim.py $c --tag wm --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  kill $wp 2>/dev/null
  log "$n 카드 $g $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200)"; up $n; }
ZG=$(claim_wait $$); log "단독 시간 카드 $ZG"
for r in s0 b1 b2 b3; do zrun $r $ZG; done
rm -f $CL/$ZG
$PY exe/wmbar_tab.py $O > $O/wmbar_table.md 2> $O/wmbar_table.err; up z_s0
# ---- 렌더 (3DGS GF wm_render, logJp 색 −0.2~0.4 공통): 판마다 완주한 a 먼저, 없으면 b, 둘 다 발산이면 더 오래 간 판 (멈춘 칸). 합성은 로컬.
W=/home/dkta/work; CD=$(pwd -P)
pick() { $PY - $O $1 <<'PYX'
import json, sys
O, v = sys.argv[1:3]
def sm(r):
    try: return json.load(open(f"{O}/{v}_{r}/summary.json"))
    except Exception: return None
ok = [r for r in "ab" if sm(r) and sm(r).get("diverged") is None]
if ok: print(f"{v}_{ok[0]} 0")
else:
    rs = [r for r in "ab" if sm(r)]
    b = max(rs, key=lambda r: sm(r).get("frames_done", 0)) if rs else None
    print(f"{v}_{b} {sm(b).get('diverged') or 0}" if b else "- 0")
PYX
}
WMM=$W/gfmodel/watermelon; [ -d $WMM/point_cloud ] || WMM=$W/gfmodel/model/watermelon; CFG=$W/wm2/gf/config.json
GTD=/tmp/dkta/ag_data/gt_wm; [ -f $GTD/meta.json ] || GTD=/dev/shm/dkta/ag_data/gt_wm
RG=$(claim_wait $$); log "렌더 카드 $RG (정답 처음 상태 $GTD $(ls $GTD 2>/dev/null | wc -l))"
if [ ! -f $GTD/meta.json ]; then mkdir -p /dev/shm/dkta/ag_data
  CUDA_VISIBLE_DEVICES=$RG $PY -u exe/pg_h5_to_pt.py --h5_dir $W/wm2/gf/simulation_ply --config $CFG --out_pt /dev/shm/dkta/ag_data/wm_rend_tmp.pt --out_dir $GTD --out_dir_n 1 --allow_nonfinite > /dev/shm/dkta/ag_data/cv_wm_rend.log 2>&1
  rm -f /dev/shm/dkta/ag_data/wm_rend_tmp.pt; fi
: > $O/video_runs.txt
for v in s0 b1 b2 b3; do read n dv < <(pick $v); echo "$v $n $dv" >> $O/video_runs.txt; [ "$n" = - ] && continue; Od=$O/$n; V=$Od/vid
  [ -s $V/lj.mp4 ] && continue; mkdir -p $V
  [ -s $Od/pred.pt ] || rclone copyto $R/$n/pred.pt $Od/pred.pt -q
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
  (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$RG $PY -u $CD/exe/wm_render.py --model_path $WMM --config $CFG --sim $Od/h5_fem --scalar_dir $Od/sc_lj --vmin -0.2 --vmax 0.4 --cmap viridis --out $V/lj.mp4 < /dev/null > $V/r_lj.log 2>&1)
  rm -rf $Od/h5_fem $Od/sc_lj $Od/pred.pt
  log "[렌더] $n $(tail -1 $V/pad.log) $(ls -la $V/lj.mp4 2>&1 | awk '{print $5}') $(tail -1 $V/r_lj.log | cut -c1-200)"
  rclone copy $V $R/$n/vid -q
done
rm -f $CL/$RG
rclone copy $O $R -q --include "video_runs.txt"
$PY exe/wmbar_tab.py $O > $O/wmbar_table.md 2> $O/wmbar_table.err
log "WMBAR_END"
