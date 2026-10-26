#!/bin/bash
# ficus 출력만 최적화 + 텐서곱 B-스플라인 FFD (사용자 10-26): (A) --interp bspline2 --cell_barrier 1.0, (B) --interp bspline3.
# 기준: lbours/ficus_v0 (선형, 커밋 110ce74) 의 명령줄(run.out 의 [BASE]·[OV]) 그대로에 위 인자만 더한다.
# 롤아웃·분석 커밋 AF_RUN_COMMIT (master). 출력 lbours/ficus_v0_bspline (처음부터 R2 통째 동기화, 제외 없음). 멱등.
# 표식: BSP_LAUNCHED, BSP_CLUSTER_DONE, BSP_FAIL
# 사용: AF_RUN_COMMIT=... AF_DETACH=1 bash exe/job_run.sh /home/dkta/work/ficus_bspline.sh <로그>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
C=${AF_RUN_COMMIT:?롤아웃 커밋}
WT=$W/wt/bspline_$C
O0=$W/lbours/ficus_v0; OGF=$W/lbours/ficus_v0_greg_fix; OLR=$W/lbours/ficus_v0_localrot; TOPO=$W/topo/pg_ficus_elastic_v0
OB=$W/lbours/ficus_v0_bspline; R=r2:storage/result/anchorflow/lbours/ficus_v0_bspline
T=F_pg_v0; DATA=$O0/data/$T; PT=$DATA/$T.pt; GTD=$O0/gt; LEN=125; mkdir -p $OB
fail(){ echo "[오류] $1"; echo "$(date '+%m-%d %H:%M') $1" >> $OB/BSP_FAIL; rclone copy $OB $R -q; exit 1; }
[ -d $WT ] || (cd $W/anchorflow && git fetch -q origin && git worktree add -q --detach $WT $C) || fail "worktree $C"
if [ -f $OB/driver.hb ] && [ $(( $(date +%s) - $(cat $OB/driver.hb) )) -lt 300 ]; then echo "[중단] 다른 드라이버가 돌고 있다 $(date)"; exit 0; fi
echo $$ > $OB/driver.pid
( while true; do date +%s > $OB/driver.hb; sleep 120; done ) & HB=$!
( while true; do rclone copy $OB $R -q; sleep 300; done ) & SYNC=$!
trap 'kill $SYNC $HB 2>/dev/null' EXIT
cd $WT || exit 1
echo "[시작] $(date) 커밋 $(git rev-parse --short HEAD) 노드 $(hostname) 잡 ${PBS_JOBID:-?}"
# 0) 단위 시험 (CPU): 새 시험 + 기존 --interp bspline/nodeavg 시험 (회귀)
if [ ! -f $OB/unit_$C.ok ]; then
  { CUDA_VISIBLE_DEVICES= python -u exe/test_interp_bspline23.py; echo "=== 기존 test_interp_smooth.py (회귀)"; CUDA_VISIBLE_DEVICES= python -u exe/test_interp_smooth.py; } > $OB/unit_$C.log 2>&1
  cat $OB/unit_$C.log
  grep -q "BSPLINE23_TEST ALL_PASS" $OB/unit_$C.log && grep -q "INTERP_SMOOTH_TEST ALL_PASS" $OB/unit_$C.log || fail "단위 시험 실패 $OB/unit_$C.log"
  touch $OB/unit_$C.ok
fi
rclone copy $OB $R -q --max-depth 1
# 1) 명령줄: ficus_v0 run.out 의 [BASE]·[OV] 그대로 + 판별 인자. 회전판 cmdline.txt 의 '선형(같은 틀)' 과도 대조
BASE=$(grep -a -m1 '^\[BASE\] ' $O0/run.out | sed 's/^\[BASE\] //'); OV=$(grep -a -m1 '^\[OV\] ' $O0/run.out | sed 's/^\[OV\] //')
[ -n "$BASE" ] && [ -n "$OV" ] || fail "ficus_v0 run.out 에서 [BASE]/[OV] 를 못 읽었다"
XA="--interp bspline2 --cell_barrier 1.0"; XB="--interp bspline3"
norm(){ tr -s ' ' <<<"$1" | sed 's/^ //; s/ $//'; }
mkargs(){ norm "--infer --data $DATA --hold_traj $T --out $1/ov --tag OVN $BASE $OV $2 --iters 0 --eval_t0 0 --eval_len $3 --tb none"; }
LRL=$(grep -m1 '^  선형(같은 틀' $OLR/cmdline.txt | sed 's/^  선형(같은 틀, --local_rot 없음): *//' | sed "s#$OLR/run#$OB/A#g")
{ echo "[명령줄] 기준 = lbours/ficus_v0 (커밋 110ce74) run.out 의 [BASE]·[OV]; 커밋 $C"
  for k in A B; do
    X=$XA; [ $k = B ] && X=$XB
    NA=$(mkargs $OB/$k "$X" $LEN); LA=$(mkargs $OB/$k "" $LEN)
    echo "  ($k) 새 인자: $NA"
    echo "  ($k) 새 환경: AF_INV_LOG=$OB/$k/inv_ov.jsonl AF_ROLL_DUMP=$OB/$k/roll_ov.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0"
    echo "  ($k) 선형 틀 대비 차이:"; diff <(tr ' ' '\n' <<<"$LA") <(tr ' ' '\n' <<<"$NA") | sed 's/^/    /'
  done
  echo "  회전판 cmdline.txt 의 선형 틀(출력 디렉토리만 바꿈) 과 이 선형 틀 대조:"
  diff <(tr ' ' '\n' <<<"$(norm "$LRL")") <(tr ' ' '\n' <<<"$(mkargs $OB/A "" $LEN)") | sed 's/^/    /'
  [ "$(norm "$LRL")" = "$(mkargs $OB/A "" $LEN)" ] && echo "  -> 같다" || echo "  -> 다르다"
} | tee $OB/cmdline.txt
[ "$(norm "$LRL")" = "$(mkargs $OB/A "" $LEN)" ] || fail "선형 틀이 회전판 cmdline 의 선형 틀과 다르다"
for k in A B; do X=$XA; [ $k = B ] && X=$XB; n=$(wc -w <<<"$X")
  [ "$(diff <(tr ' ' '\n' <<<"$(mkargs $OB/$k "" $LEN)") <(tr ' ' '\n' <<<"$(mkargs $OB/$k "$X" $LEN)") | grep -c '^>')" -eq $n ] || fail "($k) 새 인자에 다른 차이"; done
free_gpus(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<500{print $1}'; }
wait_gpu(){ local g=""; while [ -z "$g" ]; do g=$(free_gpus | head -1); [ -z "$g" ] && { date +%s > $OB/wait_gpu; sleep 120; }; done; rm -f $OB/wait_gpu; echo $g; }
# 2) 3 프레임 통합 시험: 선형 | A | B
S=$OB/smoke; mkdir -p $S
if [ ! -f $S/SMOKE_OK ]; then
  GL=($(free_gpus)); while [ ${#GL[@]} -lt 1 ]; do sleep 120; GL=($(free_gpus)); done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $S/gpu_at_start.txt
  i=0; PIDS=()
  for k in lin A B; do
    [ -f $S/$k/roll_ov.pt ] && continue
    mkdir -p $S/$k; G=${GL[$(( i % ${#GL[@]} ))]}; i=$((i+1))
    case $k in lin) X="";; A) X="$XA";; B) X="$XB";; esac
    A3=$(mkargs $S/$k "$X" 3)
    echo "AF_INV_LOG=$S/$k/inv_ov.jsonl AF_ROLL_DUMP=$S/$k/roll_ov.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py $A3" > $S/$k/cmd.txt
    echo "[시험 실행] $k GPU$G"
    env AF_INV_LOG=$S/$k/inv_ov.jsonl AF_ROLL_DUMP=$S/$k/roll_ov.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 CUDA_VISIBLE_DEVICES=$G \
      python -u exe/train_deform.py $A3 < /dev/null > $S/$k/v_ov.log 2>&1 &
    PIDS+=($!)
    [ ${#GL[@]} -lt 3 ] && wait ${PIDS[-1]}
  done
  for p_ in "${PIDS[@]}"; do wait $p_; done
  for k in lin A B; do [ -f $S/$k/roll_ov.pt ] || fail "3 프레임 시험 $k 실패 $(grep -a 'Traceback\|Error' $S/$k/v_ov.log | tail -2)"; done
  python - $S $O0/roll_ov.pt <<'PY' | tee $S/smoke.txt
import sys, json, torch
S, ref = sys.argv[1], sys.argv[2]
h = 0.056569
L = {k: torch.load(f"{S}/{k}/roll_ov.pt", map_location="cpu", weights_only=False) for k in ("lin", "A", "B")}
R0 = torch.load(ref, map_location="cpu", weights_only=False, mmap=True)
x0 = L["lin"]["x0"].float()
pot = ((x0 - torch.tensor([1., 1., 0.5])).abs() < torch.tensor([0.5, 0.5, 0.28])).all(-1)
def dmax(a, b): return float((a.float() - b.float()).norm(dim=-1).max()) / h
for t in range(3):
    mv = float((L["lin"]["pred"][t].float() - x0).norm(dim=-1).mean()) / h
    print(f"[시험] 프레임 {t+1}: 선형 평균 이동 {mv:.4e} h | 선형(이 커밋) vs ficus_v0 덤프 최대 {dmax(L['lin']['pred'][t], R0['pred'][t]):.3e} h | "
          f"A vs 선형 최대 {dmax(L['A']['pred'][t], L['lin']['pred'][t]):.3e} h | B vs 선형 최대 {dmax(L['B']['pred'][t], L['lin']['pred'][t]):.3e} h")
    for k in ("lin", "A", "B"):
        dpot = float((L[k]["pred"][t].float()[pot] - x0[pot]).norm(dim=-1).max())
        dF = float((L[k]["F"][t].float()[pot] - torch.eye(3)).abs().max())
        print(f"        {k}: 화분 상자 안 입자 {int(pot.sum())} 최대 이동 {dpot:.3e} (시뮬 단위), |F-I| 최대 {dF:.3e}")
for k in ("lin", "A", "B"):
    for ln in open(f"{S}/{k}/inv_ov.jsonl"):
        r = json.loads(ln)
        print(f"[시험 {k}] 프레임 {r['frame']}: det F_step 최소(입자) {r['det_min']:.4f}, 표본 det 최소 {r.get('smp_det_min', float('nan')):.4f} "
              f"({r.get('smp_n', 0)} 점, ≤0 {r.get('smp_det_neg', 0)}), 누적 뒤집힘 {r['inv_particles_cum']}, 상한 {r.get('cap')}")
    for ln in open(f"{S}/{k}/v_ov.log", errors="replace"):
        if "[롤아웃] F_pg_v0" in ln or "[B-스플라인" in ln or "[뒤집힘]" in ln: print(f"[시험 {k}] {ln.strip()[:240]}")
PY
  grep -q "B vs 선형" $S/smoke.txt || fail "3 프레임 시험 비교 실패"
  touch $S/SMOKE_OK
fi
rclone copy $OB $R -q
# 3) 출력만 최적화 125 프레임 (A·B 를 빈 카드 둘에 동시에; 덤프가 있으면 건너뜀)
run_full(){ local k=$1 X=$2 G=$3 O=$OB/$1; mkdir -p $O
  local NA=$(mkargs $O "$X" $LEN)
  ts=$(date +%m%d_%H%M); for f in inv_ov.jsonl v_ov.log; do [ -f $O/$f ] && mv $O/$f $O/$f.partial_$ts; done
  echo "AF_INV_LOG=$O/inv_ov.jsonl AF_ROLL_DUMP=$O/roll_ov.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py $NA" > $O/cmd.txt
  echo "[실행] $k GPU$G $(date) 커밋 $C" | tee -a $OB/BSP_LAUNCHED
  env AF_INV_LOG=$O/inv_ov.jsonl AF_ROLL_DUMP=$O/roll_ov.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 CUDA_VISIBLE_DEVICES=$G \
    python -u exe/train_deform.py $NA < /dev/null > $O/v_ov.log 2>&1
  [ -f $O/roll_ov.pt ] && echo "ov $(grep -a "롤아웃\] $T" $O/v_ov.log | tail -1 | grep -o '평균 [0-9.]*%')" > $O/err_ov.txt
}
# /home 이 100% 근처라 두 실행(~9 GB) 전에 여유 12 GB 를 기다린다
while [ $(df -BG --output=avail /home | tail -1 | tr -dc 0-9) -lt 12 ]; do echo "[디스크] /home 여유 $(df -h --output=avail /home | tail -1) -- 기다림 $(date)" | tee -a $OB/wait_disk; sleep 600; done
PIDS=()
for k in A B; do
  [ -f $OB/$k/roll_ov.pt ] && continue
  X=$XA; [ $k = B ] && X=$XB
  G=$(wait_gpu); nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $OB/gpu_at_start_$k.txt
  run_full $k "$X" $G & PIDS+=($!)
  sleep 240     # 첫 실행이 카드를 잡은 뒤 다음 빈 카드를 고른다
done
for p_ in "${PIDS[@]}"; do wait $p_; done
for k in A B; do [ -f $OB/$k/roll_ov.pt ] || fail "($k) 롤아웃 실패 $(grep -a 'Traceback\|Error' $OB/$k/v_ov.log | tail -2)"; cat $OB/$k/err_ov.txt; done
rclone copy $OB $R -q
# 3b) 렌더·탄성 색
G=$(wait_gpu)
for k in A B; do O=$OB/$k
  [ -f $O/h5_ov/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_ov.pt --traj $GTD --out $O/h5_ov > $O/h5_ov.log 2>&1
  [ -f $O/ov_ficus.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_ov --config $GTD/config.json --shape ficus --out $O/ov_ficus.mp4 > $O/r_ov.log 2>&1
  [ -f $O/ov_ficus.mp4 ] || fail "($k) 렌더 실패 $(tail -3 $O/r_ov.log)"
  PS=$O/psi; mkdir -p $PS
  [ -f $PS/psi_hist.npz ] || CUDA_VISIBLE_DEVICES=$G python -u exe/psi_color.py --gt_h5 $TOPO --dump $O/roll_ov.pt --config $GTD/config.json --out $PS \
     --eps 1e-3 --q 1 99 --range -2.9999966621398926 4.933352713584899 --cmap turbo --stats_frames 1 40 120 > $PS/psi_color.log 2>&1
  grep -q PSI_COLOR_DONE $PS/psi_color.log || fail "($k) psi 실패 $(tail -3 $PS/psi_color.log)"
  [ -f $PS/psi_ov_ficus.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_ov --config $GTD/config.json --shape ficus \
       --colors $PS/col_ov.npy --out $PS/psi_ov_ficus.mp4 > $PS/r_ov.log 2>&1
  [ -f $PS/psi_ov_ficus.mp4 ] || fail "($k) 탄성 색 렌더 실패 $(tail -3 $PS/r_ov.log)"
done
rclone copy $OB $R -q
# 4) 분석
AN=$OB/ana; mkdir -p $AN
RUNS_OSC=("선형=$O0/roll_ov.pt" "수정 Gregory=$OGF/run/roll_ov.pt" "회전판=$OLR/run/roll_ov.pt" "A=$OB/A/roll_ov.pt" "B=$OB/B/roll_ov.pt")
[ -f $AN/osc.json ] || python - $PT $AN/osc.json "${RUNS_OSC[@]}" <<'PY'
import sys, json, torch
G = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
x0 = G["x"][0].float()
pot = ((x0 - torch.tensor([1., 1., 0.5])).abs() < torch.tensor([0.5, 0.5, 0.28])).all(-1); free = ~pot
tip = x0[:, 2] >= torch.quantile(x0[free, 2], 0.99)
def ser(X):
    return {"com": (X[:, free].mean(1) - x0[free].mean(0)).tolist(), "tip": (X[:, tip].mean(1) - x0[tip].mean(0)).tolist()}
out = {"gt": ser(G["x"].float()), "n_free": int(free.sum()), "n_tip": int(tip.sum())}
for q in sys.argv[3:]:
    k, p = q.split("=", 1)
    D = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    out[k] = ser(torch.cat([x0[None], D["pred"].float()]))
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False)
print(f"[진동] 자유 입자 {int(free.sum())}, 줄기 끝 {int(tip.sum())} -> {sys.argv[2]}")
PY
[ -f $AN/osc_quick.json ] || python -u exe/ficus_osc_quick.py --gt_dir $GTD --run "lin=선형=$O0/roll_ov.pt" \
    --run "gfix=수정 Gregory=$OGF/run/roll_ov.pt" --run "rot=회전판=$OLR/run/roll_ov.pt" \
    --run "bsp2=A=$OB/A/roll_ov.pt" --run "bsp3=B=$OB/B/roll_ov.pt" --out $AN/osc_quick.json > $AN/osc_quick.log 2>&1
grep -q OSC_DONE $AN/osc_quick.log || fail "빠른 진동 분석 실패 $(tail -3 $AN/osc_quick.log)"
EG=$AN/energy; mkdir -p $EG
if ! grep -q FICUS_ENERGY_OK $EG/energy.log 2>/dev/null; then
  python -u exe/ficus_energy_cmp.py --gt_pt $PT --gt_psi $O0/psi/psi_gt.npy --run "선형=$O0" --run "수정 Gregory=$OGF/run" --run "회전판=$OLR/run" \
     --run "A=$OB/A" --run "B=$OB/B" --out_json $EG/energy.json --fig $EG/ficus_bspline_energy.png --title "ficus" > $EG/energy.log 2>&1; cat $EG/energy.log
  grep -q FICUS_ENERGY_OK $EG/energy.log || fail "에너지 실패"
fi
# 4b) det F (누적·스텝)·위치 오차: 덤프 + inv_ov.jsonl
[ -f $AN/detstats.json ] || python - $AN/detstats.json $PT "선형=$O0" "수정 Gregory=$OGF/run" "회전판=$OLR/run" "A=$OB/A" "B=$OB/B" <<'PY' | tee $AN/detstats.txt
import sys, json, torch
G = torch.load(sys.argv[2], map_location="cpu", weights_only=False, mmap=True)
gx = G["x"].float(); h = 0.056569
out = {}
for q in sys.argv[3:]:
    k, d = q.split("=", 1)
    D = torch.load(d + "/roll_ov.pt", map_location="cpu", weights_only=False, mmap=True)
    T = D["pred"].shape[0]
    dmin, dneg, err = [], [], []
    for t in range(T):
        dt_ = torch.linalg.det(D["F"][t].double())
        dmin.append(float(dt_.min())); dneg.append(int((dt_ <= 0).sum()))
        err.append(float((D["pred"][t].float() - gx[t + 1]).norm(dim=-1).mean()) / h)
    inv = [json.loads(l) for l in open(d + "/inv_ov.jsonl")]
    e = open(d + "/err_ov.txt").read().strip() if __import__("os").path.exists(d + "/err_ov.txt") else ""
    out[k] = dict(detF_min=dmin, detF_neg=dneg, err_h=err, err_ov=e,
                  step_det_min=[r["det_min"] for r in inv], step_inv=[r["inv_particles_step"] for r in inv],
                  smp_det_min=[r.get("smp_det_min") for r in inv], smp_det_neg=[r.get("smp_det_neg") for r in inv],
                  cap=[r.get("cap") for r in inv])
    sm = [x for x in out[k]["smp_det_min"] if x is not None]
    print(f"[det] {k}: 누적 det F 최소 {min(dmin):.4f}, 뒤집힌 입자-프레임 {sum(dneg)}; 스텝 det 최소 {min(out[k]['step_det_min']):.4f}, "
          f"스텝 뒤집힘 합 {sum(out[k]['step_inv'])}; 표본 det 최소 {min(sm) if sm else float('nan'):.4f}; 평균 위치 오차 {sum(err)/len(err):.4f} h; {e}")
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False)
PY
[ -f $AN/detstats.json ] || fail "det 통계 실패"
rclone copy $OB $R -q
# 4c) 셀별 잠긴 에너지 (셀 = 기존 Kuhn 사면체; 정답·선형·수정 Gregory·회전판은 ficus_v0_localrot/ana/locked 결과 복사 재사용, 원본 손대지 않음)
LK=$AN/locked; mkdir -p $LK; LO=$OLR/ana/locked
if [ ! -f $LK/LOCKED_BSP_DONE ]; then
  G=$(wait_gpu)
  for k in gt lin gfix rot; do for f in cells_$k.npz dens_$k.npy; do [ -f $LK/$f ] || cp $LO/$f $LK/$f; done; done
  [ -f $LK/summary.json ] || python -c "import json;s=json.load(open('$LO/summary.json'));json.dump({k:s[k] for k in ('gt','lin','gfix','rot')},open('$LK/summary.json','w'),ensure_ascii=False,indent=0)"
  CUDA_VISIBLE_DEVICES=$G python -u exe/cell_locked_energy.py --config $GTD/config.json --vol_pt $PT --node_h 0.056569 --out $LK \
    --traj "gt:정답 (PG):#222222:h5:$TOPO" --traj "lin:선형:#d9480f:dump:$O0/roll_ov.pt" \
    --traj "gfix:수정 Gregory:#1c7ed6:dump:$OGF/run/roll_ov.pt" --traj "rot:회전판:#2b8a3e:dump:$OLR/run/roll_ov.pt" \
    --traj "bsp2:A 2차+장벽:#e8590c:dump:$OB/A/roll_ov.pt" --traj "bsp3:B 3차+차 상한:#7048e8:dump:$OB/B/roll_ov.pt" \
    --energy_json $EG/energy.json --energy_keys "gt=정답 (PG),lin=선형,gfix=수정 Gregory,rot=회전판,bsp2=A,bsp3=B" \
    --fig $LK/ficus_bspline_locked.png --table $LK/ficus_bspline_locked.md < /dev/null > $LK/run.log 2>&1
  grep -q LOCKED_CALC_DONE $LK/run.log || fail "잠긴 에너지 실패 $(grep -av 'it/s\|s/it' $LK/run.log | tail -3)"
  touch $LK/LOCKED_BSP_DONE
fi
grep -a "재사용\|검산\|수렴" $LK/run.log | head -12
kill $SYNC 2>/dev/null; rclone copy $OB $R -q
touch $OB/BSP_CLUSTER_DONE; rclone copy $OB $R -q; echo "BSP_CLUSTER_DONE $(date)"
