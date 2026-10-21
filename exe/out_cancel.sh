#!/bin/bash
# lego --net_inertial 롤아웃 "멈춤" 진단 (사용자 10-21): 노드 변위 = u₀ + δ 에서 멈출 때 δ ≈ −u₀ 인가.
#   학습 없음. 기존 롤아웃과 같은 명령줄(spfloor.sh roll old_multi_*, spstick.sh roll m_multi_*) + AF_DELTA_DUMP 로 다시 굴려
#   프레임별 노드 u₀·dpn(=u₀+δ)·위치·질량·바닥 거리를 덤프 -> 오차·pred 가 기존과 같은지 확인 -> exe/out_cancel_viz.py.
# 사용: AF_DETACH=1 bash exe/job_run.sh /home/dkta/work/out_cancel.sh /home/dkta/work/lbours/out_cancel/run.log
# 출력은 처음부터 R2 lbours/out_cancel 과 통째 동기화 (제외 없음). 표식: 로그 끝 OUTCANCEL_DONE / OUTCANCEL_FAIL
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
C=${OC_COMMIT:?커밋}; WT=$W/wt/outcancel_$C
[ -d $WT ] || (cd $W/anchorflow && git fetch -q origin && git worktree add -q --detach $WT $C) || { echo "worktree 실패"; echo OUTCANCEL_FAIL; exit 1; }
P=$W/lbpool; U=$W/lbours/out_cancel; R=r2:storage/result/anchorflow/lbours/out_cancel; mkdir -p $U
RRES=r2:storage/result/anchorflow
cd $WT || { echo OUTCANCEL_FAIL; exit 1; }
echo "[시작] $(date) 커밋 $(git rev-parse --short HEAD) 노드 $(hostname)"
( while true; do rclone copy $U $R -q; sleep 300; done ) & SYNC=$!
fail(){ echo "[실패] $*"; kill $SYNC 2>/dev/null; rclone copy $U $R -q; echo OUTCANCEL_FAIL; exit 1; }
# ---- 기존 실행과 같은 플래그 (spfloor.sh / spstick.sh 원문 그대로) ----
COMMON="--no_mat --arch sgnn --gnn_layers 8 --hidden 128 --obj pts --net_inertial"
REST_OLD="--inv_barrier 1.0 --inv_jhat 0.3 --inv_ext linear --node_h 0.028284 --dt_cond --dt_scale --v0_traj --det_eps 0.1 --det_w 100 --lr 3e-4 --batch 8 --det_every 200 --pool_fill $P/fill --pool_cfg $P/cfg --pool_combos lego_jelly --pool_drop 0.0,0.02,0.0,1.5 --pool_frames 80 --phase2 --phys_w 1.0 --save_every 100000"
FLOOR_OLD="--bc_level particle --bc_mode barrier --bc_kappa 1.0 --bc_ext linear --bc_mpm"
FLOOR_SP="--bc_level node --bc_mode project --bc_softplus 0.01"
REST=$(echo "$REST_OLD" | sed -e 's/--inv_barrier 1.0 --inv_jhat 0.3 --inv_ext linear/--inv_barrier 0/' -e 's/--det_w 100/--det_w 0 --det_keep_elastic/')
BASE_OLD="$COMMON $FLOOR_OLD $REST_OLD"
BASE_ST="$COMMON $FLOOR_SP $REST --bc_stick 0.01 --bc_stick_xn"
echo "[BASE_OLD] $BASE_OLD"; echo "[BASE_ST] $BASE_ST"
SPF=$W/lbours/spfloor; SPS=$W/lbours/spstick
CK_OLD=$SPF/ck_old_multi.pt; CK_ST=$SPS/multi/ck.pt
[ -f $CK_OLD ] && [ -f $CK_ST ] || fail "체크포인트 없음"
gt_of(){ [ $1 = L000 ] && echo $W/lbdata/test_low/L000 || echo $W/lbdata/test/$1; }
declare -A REF=( [oldmulti_0004]=23.243 [oldmulti_L000]=25.710 [spstick_0004]=25.568 [spstick_L000]=12.208 )
declare -A OLDV=( [oldmulti_0004]=$SPF/v_old_multi_0004 [oldmulti_L000]=$SPF/v_old_multi_L000 [spstick_0004]=$SPS/v_m_multi_0004 [spstick_L000]=$SPS/v_m_multi_L000 )
declare -A OLDD=( [oldmulti_0004]=$SPF/roll_old_multi_0004.pt [oldmulti_L000]=$SPF/roll_old_multi_L000.pt [spstick_0004]=$SPS/roll_m_multi_0004.pt [spstick_L000]=$SPS/roll_m_multi_L000.pt )
run(){ N=$1; T=$2; g=$3; CK=$4; B=$5
  echo "[롤아웃 시작] $N GPU$g $(date +%T)"
  if [ ! -f $U/delta_$N.pt ]; then
    rm -f $U/inv_$N.jsonl $U/tun_$N.jsonl
    echo "AF_DELTA_DUMP=$U/delta_$N.pt AF_ROLL_DUMP=$U/roll_$N.pt AF_ROLL_DUMP_F=1 AF_ROLL_TAG=$T AF_ROLL_T0=0 AF_ROLL_NOGRAD=1 python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $U/v_$N --tag EV --resume $CK $B --iters 0 --eval_t0 0 --eval_len 80 --tb none" > $U/cmd_$N.txt
    env AF_INV_LOG=$U/inv_$N.jsonl AF_TUNNEL_LOG=$U/tun_$N.jsonl AF_DELTA_DUMP=$U/delta_$N.pt \
      AF_ROLL_DUMP=$U/roll_$N.pt AF_ROLL_DUMP_F=1 AF_ROLL_TAG=$T AF_ROLL_T0=0 AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$g \
      python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $U/v_$N --tag EV --resume $CK $B \
      --iters 0 --eval_t0 0 --eval_len 80 --tb none > $U/v_$N.log 2>&1
    ( rclone copyto $U/delta_$N.pt $R/delta_$N.pt -q; rclone copyto $U/roll_$N.pt $R/roll_$N.pt -q ) &
  fi
  grep -a "롤아웃\] $T\|δ 덤프\|Traceback\|Error" $U/v_$N.log | tail -3 | cut -c1-200
  echo "[롤아웃 끝] $N $(date +%T)"; touch $U/.done_$N; }
# ---- 그림 스크립트 자체 시험 (가짜 3 프레임, CPU) -- 롤아웃 전에 그림 코드 오류를 잡는다 ----
ST=$U/selftest; mkdir -p $ST
CUDA_VISIBLE_DEVICES= python - <<EOF > $ST/gen.log 2>&1
import torch
g = torch.Generator().manual_seed(0)
for n in ("a_0004", "b_0004"):
    fr = []
    for i in range(3):
        M = 500
        u0 = torch.randn(M, 3, generator=g) * 0.01; de = -u0 * (i / 2) + 0.001 * torch.randn(M, 3, generator=g)
        fr.append((torch.rand(M, 3, generator=g), u0, u0 + de, torch.rand(M, generator=g), torch.rand(M, generator=g) * 0.1, torch.rand(M, generator=g) * 0.05))
    torch.save({"frames": fr, "EXT": 1.3, "h": 1 / 60}, "$ST/d_" + n + ".pt")
    x0 = torch.rand(100, 3, generator=g)
    torch.save({"pred": x0[None] + 0.01 * torch.rand(3, 100, 3, generator=g), "gt": x0[None] + 0.01 * torch.rand(3, 100, 3, generator=g), "x0": x0}, "$ST/r_" + n + ".pt")
print("GEN_OK")
EOF
CUDA_VISIBLE_DEVICES= python -u exe/out_cancel_viz.py --case "a_0004=가짜 가 · 0004=$ST/d_a_0004.pt=$ST/r_a_0004.pt=$SPF/old_multi_0004.mp4=$SPF/gt_0004.mp4=1" \
  --case "b_0004=가짜 나 · 0004=$ST/d_b_0004.pt=$ST/r_b_0004.pt=$ST/없음.mp4=$ST/없음.mp4=1" \
  --out_vid $ST --out_fig $ST --json $ST/t.json > $ST/viz.log 2>&1
tail -3 $ST/viz.log
grep -q OUTCANCEL_VIZ_OK $ST/viz.log || fail "그림 스크립트 자체 시험 실패 ($ST/viz.log)"
# 빈 GPU 만 (메모리 < 1 GB, 이 노드는 다른 실행도 돌고 있다)
mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<1024{print $1}')
echo "[빈 GPU] ${FREE[*]}"
[ ${#FREE[@]} -ge 1 ] || fail "빈 GPU 없음"
JOBS=("oldmulti_0004 0004 $CK_OLD BASE_OLD" "oldmulti_L000 L000 $CK_OLD BASE_OLD" "spstick_0004 0004 $CK_ST BASE_ST" "spstick_L000 L000 $CK_ST BASE_ST")
k=0; PIDS=()
for j in "${JOBS[@]}"; do
  read -r N T CK BV <<<"$j"
  [ -f $U/.done_$N ] && [ -f $U/delta_$N.pt ] && continue
  g=${FREE[$((k % ${#FREE[@]}))]}; k=$((k + 1))
  if [ $k -gt ${#FREE[@]} ]; then wait "${PIDS[0]}"; PIDS=("${PIDS[@]:1}"); fi
  run $N $T $g $CK "${!BV}" >> $U/task_$N.log 2>&1 & PIDS+=($!); sleep 20
done
wait "${PIDS[@]}"
for N in oldmulti_0004 oldmulti_L000 spstick_0004 spstick_L000; do cat $U/task_$N.log; done
# ---- 재현 확인: 오차 (로그 3 자리) + 기존 실행과 인자 (경로 외) + pred 최대 차 ----
python - <<EOF > $U/repro.txt 2>&1
import json, re, torch
ref = {"oldmulti_0004": 23.243, "oldmulti_L000": 25.710, "spstick_0004": 25.568, "spstick_L000": 12.208}
oldv = {"oldmulti_0004": "${OLDV[oldmulti_0004]}", "oldmulti_L000": "${OLDV[oldmulti_L000]}", "spstick_0004": "${OLDV[spstick_0004]}", "spstick_L000": "${OLDV[spstick_L000]}"}
oldd = {"oldmulti_0004": "${OLDD[oldmulti_0004]}", "oldmulti_L000": "${OLDD[oldmulti_L000]}", "spstick_0004": "${OLDD[spstick_0004]}", "spstick_L000": "${OLDD[spstick_L000]}"}
ok = True
for n, r in ref.items():
    lg = open(f"$U/v_{n}.log", encoding="utf-8", errors="replace").read()
    m = re.findall(r"롤아웃\] \S+.*?평균 ([0-9.]+)%", lg)
    e = float(m[-1]) if m else float("nan")
    na = json.load(open(f"$U/v_{n}/EV.json"))["args"]; oa = json.load(open(f"{oldv[n]}/EV.json"))["args"]
    diff = {k: (oa.get(k), na.get(k)) for k in set(oa) | set(na) if k not in ("out", "resume") and oa.get(k) != na.get(k)}
    P = torch.load(f"$U/roll_{n}.pt", map_location="cpu", weights_only=False, mmap=True)["pred"]
    Q = torch.load(oldd[n], map_location="cpu", weights_only=False, mmap=True)["pred"]
    dmax = float((P.float() - Q.float()).abs().max()) if P.shape == Q.shape else float("nan")
    same = abs(e - r) < 0.0015
    ok &= same
    print(f"[재현] {n}: 새 {e:.3f}% 기존 {r:.3f}% {'같음' if same else '다름'} | pred 최대 차 {dmax:.3e} | 기존과 다른 인자 {diff}")
print("REPRO_OK" if ok else "REPRO_DIFF")
EOF
cat $U/repro.txt; rclone copy $U $R -q
grep -q REPRO_OK $U/repro.txt || fail "재현 오차가 기존과 다름 -- 그림 안 그림 (repro.txt)"
# ---- 그림·영상 (CPU) ----
mkdir -p $U/v20261009 $U/ovnet
CASES=()
for x in "oldmulti_0004=이전 바닥(장벽·벌점) 다섯 궤적 · 0004=$SPF/old_multi_0004.mp4=$SPF/gt_0004.mp4=10" \
         "oldmulti_L000=이전 바닥(장벽·벌점) 다섯 궤적 · L000=$SPF/old_multi_L000.mp4=$SPF/gt_L000.mp4=16" \
         "spstick_0004=새 바닥(softplus+미끄러짐 방지) 다섯 궤적 · 0004=$SPS/m_multi_0004.mp4=$SPS/gt_0004.mp4=10" \
         "spstick_L000=새 바닥(softplus+미끄러짐 방지) 다섯 궤적 · L000=$SPS/m_multi_L000.mp4=$SPS/gt_L000.mp4=16"; do
  N=${x%%=*}; rest=${x#*=}; TI=${rest%%=*}; rest=${rest#*=}
  CASES+=(--case "$N=$TI=$U/delta_$N.pt=$U/roll_$N.pt=$rest"); done
CUDA_VISIBLE_DEVICES= python -u exe/out_cancel_viz.py "${CASES[@]}" --out_vid $U/v20261009 --out_fig $U/ovnet --json $U/out_cancel.json > $U/viz.log 2>&1
tail -5 $U/viz.log
grep -q OUTCANCEL_VIZ_OK $U/viz.log || fail "그림 실패 (viz.log)"
for N in oldmulti_0004 oldmulti_L000 spstick_0004 spstick_L000; do rclone copyto $U/v20261009/lego_out_sphere_$N.mp4 $RRES/v20261009/lego_out_sphere_$N.mp4 -q; done
rclone copyto $U/ovnet/lego_out_cancel_summary.png $RRES/ovnet/lego_out_cancel_summary.png -q
rclone copyto $U/ovnet/lego_out_cancel.md $RRES/ovnet/lego_out_cancel.md -q
kill $SYNC 2>/dev/null
rclone copy $U $R -q && echo "[R2 업로드 완료] $(date)" && touch $U/ALL_DONE && rclone copy $U $R -q
echo OUTCANCEL_DONE
