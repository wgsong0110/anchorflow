#!/bin/bash
# 절 CM23 묶음 k1new: 새 기본 판 (장벽 바닥 (B) + F·재질 상태 가우스 점 보간 교체 --fp_interp; 수박 = 절 CM21 (3)) 을 프레임당 1 걸음 (K 1 = 기본 판, 사용자 결정 11-07) 으로 전 씬.
#   비교 = 같은 판 예전 걸음 수 (레고·ficus·wolf·bread K 2, 수박 K 3) -- 있는 결과는 재사용 (go 스크립트가 R2 에서 받아 둠: 레고 K2 = fint K2_on, 수박 K3 = wmbar b3_a·b3_b, wolf K2 = wolfnew b_on 이 있으면).
#   반복 상한 (뉴턴·PCG) 은 씬마다 기본 판 값 그대로 (K 만 바꿈): 레고 2x4, ficus 2x24, wolf 2x5, bread 2x2, 수박 2x5.
#   판정 판 = 결정적 (--det) + 결과 불변 진단, 수박만 비결정 두 번 (a·b). 단독 시간 = 진단·덤프·--det 뺀 비결정 판, 카드 하나에 차례로 (씬마다 K1·비교 번갈아 두 번).
#   수박 K1 은 --dump --dump_lj (손상 상위 표시 렌더용, 결과 불변).
# 잡 안, 커밋 디렉토리에서; 출력 $1, R2 $2. 판이 끝나면 바로 R2.
O=$1; R=$2; mkdir -p $O
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8
source exe/fem_cm2_scn.sh                                                                     # SD·AG·FI·cm_data (자료만 씀)
for t in L000 ficus wolf bread; do cm_data $t; done
WMD=/tmp/dkta/ag_data; mkdir -p $WMD
[ -s $WMD/wm.pt ] || { rclone copyto r2:storage/result/anchorflow/lbours/fem_ag_data/wm.pt $WMD/wm.pt.$$ -q && mv $WMD/wm.pt.$$ $WMD/wm.pt; }
BA0="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --mg_wreuse --mg_lreuse --mv2 --lsfast --fk --gq_oct 0.1 --mgbar 2 --ga_y0ut --hspec --ncomp --ga hht --rho 0.8"
FBB="--floor vel --fl_bar --fl_bst 1 --y0eb"
TL="--tolN 1e-2 --tolP 1e-1"
# 씬 명령 (K 와 진단 빼고) -- 절 CM20 fint K2·절 CM8 cm8j o_ficus·o_bread·b5_wolf·절 CM21 wmbar b3 명령 그대로
base_of() { case $1 in
  L) echo "exe/fem_an_rt_sim.py --data $SD/L000.pt --frames 80 $FBB $BA0 --newton 2 --cg 4 $TL --fp_interp --tag L000";;
  F) echo "exe/fem_an_rt_sim.py --data $FI --frames 125 --floor none --cfg_bc --region_density --pg_vdamp $BA0 --newton 2 --cg 24 $TL --fp_interp --tag ficus";;
  W) echo "exe/fem_an_rt_sim.py --data $AG/wolf.pt --frames 50 $FBB --pbc --mlow 1e-3 $BA0 --cell 2 --newton 2 --cg 5 $TL --fp_interp --tag wolf";;
  B) echo "exe/fem_an_rt_sim.py --data $AG/bread.pt --frames 300 --floor none --pbc --pg_vdamp --visc_rpic --mlow 1e-3 $BA0 --newton 2 --cg 2 $TL --fp_interp --tag bread";;
  M) echo "exe/fem_ao_rt_sim.py --data $WMD/wm.pt --frames 100 --floor vel --floor_plane 1,1,0.02,0,0,1 --cell 1.5 --mlow 1e-3 --integ mid --ga hht --rho 0.7 --ga_y0ut $TL --newton 2 --cg 5 --ls_inv --y0u --y0l --fq_rl --ls_end --gq_mass --elx snh --fl_bar --fl_bst 1 --y0eb --lj_interp --fp_interp --tag wm";;
esac; }
dg_of() { case $1 in                                                                           # 판정 판 진단 (결과 불변)
  L) echo "--det --pdiag --qjd --fhash";;
  F) echo "--det --pdiag --qjd --fhash --dump";;
  W) echo "--det --hgt --bar_cnt --pdiag --qjd --fhash";;
  B) echo "--det --pdiag --qjd --fhash";;
  M) echo "--gq_chk --nan_step --jdiag --bar_cnt --qjd --dump --dump_lj";;
esac; }
log() { echo "[$(date '+%F %T')] $*" >> $O/k1new.log; rclone copyto $O/k1new.log $R/k1new.log -q; }
up() { rclone copy $O/$1 $R/$1 -q --transfers 8; }
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
run() {    # run 이름 씬 K 카드 [frames]
  local n=$1 s=$2 k=$3 g=$4 nf=${5:-} c
  [ -f $O/$n/summary.json ] && return
  c="$(base_of $s) --K $k $(dg_of $s)"; [ -n "$nf" ] && c=$(echo "$c" | sed "s/--frames [0-9]*/--frames $nf/")
  rm -rf $O/$n; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/apps_at_start.txt 2>&1
  log "$n 카드 $g 시작"
  CUDA_VISIBLE_DEVICES=$g timeout 5400 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?
  [ $s = F ] && [ -f $O/$n/pred.pt ] && $PY exe/fem_p_period.py --data $FI --pred $O/$n/pred.pt > $O/$n/period.txt 2>&1
  log "$n rc=$rc $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(tail -2 $O/$n/period.txt 2>/dev/null | tr '\n' ' ') $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  up $n; }
# ---- 연기 (K1 3 프레임, 씬마다): 컴파일·실행·요약 있음·발산 없음
SM=$O/smoke; mkdir -p $SM
if [ ! -f $SM/ok ]; then
  for s in L F W B M; do ( g=$(claim_wait $BASHPID); c="$(base_of $s | sed 's/--frames [0-9]*/--frames 3/') --K 1 $(dg_of $s | sed 's/--dump_lj//; s/--dump//')"
    mkdir -p $SM/$s; echo "$PY -u $c --out $SM/$s" > $SM/$s/cmd.txt
    CUDA_VISIBLE_DEVICES=$g timeout 1800 $PY -u $c --out $SM/$s < /dev/null > $SM/$s/log.txt 2>&1; echo "rc=$? 카드 $g" > $SM/$s/rc.txt; rm -f $CL/$g ) & sleep 3; done
  wait
  OK=1; MSG=""
  for s in L F W B M; do
    v=$($PY -c "import json; s=json.load(open('$SM/$s/summary.json')); print(int(s['frames_done'] == 3 and not s.get('diverged')))" 2>/dev/null || echo 0)
    [ "$v" = 1 ] || OK=0; MSG="$MSG $s:$v $(cat $SM/$s/rc.txt) $(grep -a 'rror\|Trace' $SM/$s/log.txt | tail -2 | cut -c1-200 | tr '\n' ' ')"; done
  log "[연기] OK=$OK$MSG"; rclone copy $SM $R/smoke -q
  [ $OK = 1 ] || { echo "연기 실패 $(date '+%F %T')" > $O/k1new.FAIL; exit 1; }
  touch $SM/ok
fi
# ---- 판정 판: 공용 대기열 (긴 판 먼저), 줄마다 카드 하나
Q=$O/queue.txt; [ -f $Q ] || printf '%s\n' "M_K1_a M 1" "M_K1_b M 1" "B_K1 B 1" "B_K2 B 2" "F_K1 F 1" "F_K2 F 2" "W_K1 W 1" "L_K1 L 1" "L_K2 L 2" "W_K2 W 2" > $Q
echo 0 > $O/qpos
lane() { local me=$BASHPID i ln g
  while :; do
    i=$(flock $O/qpos.lock bash -c "i=\$(cat $O/qpos); echo \$((i + 1)) > $O/qpos; echo \$i")
    ln=$(sed -n "$((i + 1))p" $Q); [ -n "$ln" ] || break
    set -- $ln; [ -f $O/$1/summary.json ] && continue
    if [ $1 = W_K2 ]; then                                                                     # wolf K2 는 묶음 wolfnew b_on (같은 명령 + 덤프) 이 있으면 재사용
      rclone copy r2:storage/result/anchorflow/lbours/wolfnew/b_on $O/W_K2 -q --exclude "pred.pt" --exclude "*.h5" 2>/dev/null
      [ -f $O/W_K2/summary.json ] && { echo "재사용 R2 lbours/wolfnew/b_on" > $O/W_K2/reuse.txt; log "W_K2 = wolfnew b_on 재사용"; up W_K2; continue; }
      rm -rf $O/W_K2; fi
    g=$(claim_wait $me)
    run $1 $2 $3 $g
    rm -f $CL/$g
  done; }
for k in 1 2 3 4 5 6 7; do lane & sleep 2; done
wait
log "판정 판 끝"
$PY exe/k1new_tab.py $O > $O/k1new_table.md 2> $O/k1new_table.err; rclone copy $O $R -q --include "k1new_table.*"
# ---- 단독 시간 (카드 하나 잡고 차례로; 비결정, 진단·덤프 없음; 씬마다 K1 a · 비교 a · K1 b · 비교 b; 판마다 그 카드·노드 계산 프로세스 수 10 초마다)
zrun() { local n=$1 s=$2 k=$3 g=$4 c
  [ -f $O/$n/summary.json ] && return
  c="$(base_of $s) --K $k"; mkdir -p $O/$n; echo "$PY -u $c --out $O/$n" > $O/$n/cmd.txt
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > $O/$n/gpu_before.txt 2>&1
  ( while sleep 10; do echo "$(date '+%T') 카드 $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .) 노드 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" >> $O/$n/gpu_procs.txt; done ) & local wp=$!
  CUDA_VISIBLE_DEVICES=$g timeout 3600 $PY -u $c --out $O/$n < /dev/null > $O/$n/log.txt 2>&1
  kill $wp 2>/dev/null
  log "$n 카드 $g $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200)"; up $n; }
ZG=$(claim_wait $$); log "단독 시간 카드 $ZG"
for s in L F W B M; do kd=2; [ $s = M ] && kd=3
  for r in a b; do zrun z_${s}_K1_$r $s 1 $ZG; zrun z_${s}_K${kd}_$r $s $kd $ZG; done; done
rm -f $CL/$ZG
$PY exe/k1new_tab.py $O > $O/k1new_table.md 2> $O/k1new_table.err; rclone copy $O $R -q --include "k1new_table.*"
# ---- 수박 손상 상위 표시 렌더 (사용자 요청 11-07): 정답 (원래 정답 h5 위치·F + GF 다시 돌린 logJp, 절 CM13.4 wmJd gt_jp) | 우리 K1 (완주한 a 먼저)
W=/home/dkta/work; CD=$(pwd -P)
WMM=$W/gfmodel/watermelon; [ -d $WMM/point_cloud ] || WMM=$W/gfmodel/model/watermelon; CFG=$W/wm2/gf/config.json; GTH=$W/wm2/gf/simulation_ply
GTD=/tmp/dkta/ag_data/gt_wm; [ -f $GTD/meta.json ] || GTD=/dev/shm/dkta/ag_data/gt_wm
RG=$(claim_wait $$); log "렌더 카드 $RG"
if [ ! -f $GTD/meta.json ]; then mkdir -p /dev/shm/dkta/ag_data
  CUDA_VISIBLE_DEVICES=$RG $PY -u exe/pg_h5_to_pt.py --h5_dir $GTH --config $CFG --out_pt /dev/shm/dkta/ag_data/wm_rend_tmp.pt --out_dir $GTD --out_dir_n 1 --allow_nonfinite > /dev/shm/dkta/ag_data/cv_wm_rend.log 2>&1
  rm -f /dev/shm/dkta/ag_data/wm_rend_tmp.pt; fi
n=M_K1_a; $PY -c "import json,sys; s=json.load(open('$O/M_K1_a/summary.json')); sys.exit(0 if not s.get('diverged') else 1)" 2>/dev/null || n=M_K1_b
V=$O/vid; mkdir -p $V; echo $n > $V/run.txt; Od=$O/$n
GJ=$O/gt_jp; [ -f $GJ/sc_0100.npy ] || rclone copy r2:storage/result/anchorflow/lbours/fem_cm/wmJd/gt_jp $GJ -q
$PY exe/wm_dmgtop.py --gt_jp $GJ --out_gt $V/cat_gt --ours $Od --out_ours $V/cat_ours --out_ours_gtthr $V/cat_ours_gtthr --json $V/dmgtop.json > $V/dmgtop.log 2>&1
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
rw() { (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$RG $PY -u $CD/exe/wm_render.py --model_path $WMM --config $CFG --frames 100 "$@" < /dev/null); }
rw --sim $GTH --cat_dir $V/cat_gt --out $V/gt_top.mp4 > $V/r_gt_top.log 2>&1
rw --sim $Od/h5_fem --cat_dir $V/cat_ours --out $V/ours_top.mp4 > $V/r_ours_top.log 2>&1
rw --sim $Od/h5_fem --cat_dir $V/cat_ours_gtthr --out $V/ours_gtthr.mp4 > $V/r_ours_gtthr.log 2>&1
rw --sim $GTH --out $V/gt_plain.mp4 > $V/r_gt_plain.log 2>&1
rm -rf $Od/h5_fem $V/cat_gt $V/cat_ours $V/cat_ours_gtthr
log "[렌더] $n $(tail -1 $V/pad.log) $(tail -3 $V/dmgtop.log | tr '\n' ' ') $(ls -la $V/*.mp4 2>&1 | awk '{print $5, $9}' | tr '\n' ' ') $(tail -1 $V/r_gt_top.log | cut -c1-150) $(tail -1 $V/r_ours_top.log | cut -c1-150)"
rclone copy $V $R/vid -q
for m in M_K1_a M_K1_b; do [ -s $O/$m/pred.pt ] && rclone check $O/$m $R/$m --one-way --size-only -q --include pred.pt && rm -f $O/$m/pred.pt; done   # /dev/shm 자리: R2 확인 뒤만
rm -f $CL/$RG
$PY exe/k1new_tab.py $O > $O/k1new_table.md 2> $O/k1new_table.err
log "K1NEW_END"
