#!/bin/bash
# ficus 출력만 최적화 + 셀 안 버블 회전 (--local_rot bubble, 사용자 10-24).
# 기준: lbours/ficus_v0 (선형, 커밋 110ce74) 의 명령줄(run.out 의 [BASE]·[OV]) 그대로에 --local_rot bubble 만 더한다.
# 롤아웃 커밋 AF_RUN_COMMIT (localrot-f599abb: f599abb = 수정 Gregory 실행 코드 + localrot 만), 분석 커밋 AF_ANA_COMMIT (master).
# 출력 lbours/ficus_v0_localrot (처음부터 R2 통째 동기화, 제외 없음). 멱등: 다시 띄우면 끝난 단계는 건너뛴다.
# 표식: LOCALROT_LAUNCHED, LOCALROT_CLUSTER_DONE, LOCALROT_FAIL
# 사용: AF_RUN_COMMIT=... AF_ANA_COMMIT=... AF_DETACH=1 bash exe/job_run.sh /home/dkta/work/ficus_localrot.sh <로그>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
C=${AF_RUN_COMMIT:?롤아웃 커밋}; CA=${AF_ANA_COMMIT:?분석 커밋}
WT=$W/wt/localrot_$C; WA=$W/wt/localrot_ana_$CA
O0=$W/lbours/ficus_v0; OGF=$W/lbours/ficus_v0_greg_fix; TOPO=$W/topo/pg_ficus_elastic_v0
OB=$W/lbours/ficus_v0_localrot; R=r2:storage/result/anchorflow/lbours/ficus_v0_localrot
T=F_pg_v0; DATA=$O0/data/$T; PT=$DATA/$T.pt; GTD=$O0/gt; LEN=125; mkdir -p $OB
fail(){ echo "[오류] $1"; echo "$(date '+%m-%d %H:%M') $1" >> $OB/LOCALROT_FAIL; rclone copy $OB $R -q; exit 1; }
for p in "$WT $C" "$WA $CA"; do set -- $p
  [ -d $1 ] || (cd $W/anchorflow && git fetch -q origin && git worktree add -q --detach $1 $2) || fail "worktree $2"; done
if [ -f $OB/driver.hb ] && [ $(( $(date +%s) - $(cat $OB/driver.hb) )) -lt 300 ]; then echo "[중단] 다른 드라이버가 돌고 있다 $(date)"; exit 0; fi
echo $$ > $OB/driver.pid
( while true; do date +%s > $OB/driver.hb; sleep 120; done ) & HB=$!
( while true; do rclone copy $OB $R -q; sleep 300; done ) & SYNC=$!
trap 'kill $SYNC $HB 2>/dev/null' EXIT
cd $WT || exit 1
echo "[시작] $(date) 롤아웃 커밋 $(git rev-parse --short HEAD) 분석 커밋 $CA 노드 $(hostname) 잡 ${PBS_JOBID:-?}"
git diff --stat f599abb HEAD | tee $OB/code_diff_vs_f599abb.txt
# 0) 단위 시험 (CPU): 실제 ficus 입자 + 무작위 입자
if [ ! -f $OB/unit_$C.ok ]; then
  { CUDA_VISIBLE_DEVICES= python -u exe/test_local_rot.py --pt $PT; echo "=== 무작위 입자"; CUDA_VISIBLE_DEVICES= python -u exe/test_local_rot.py; } > $OB/unit_$C.log 2>&1
  cat $OB/unit_$C.log
  [ $(grep -c "\[전체\] 통과" $OB/unit_$C.log) -eq 2 ] || fail "단위 시험 실패 $OB/unit_$C.log"
  touch $OB/unit_$C.ok
fi
rclone copy $OB $R -q --max-depth 1
# 1) 명령줄: ficus_v0 run.out 의 [BASE]·[OV] 그대로 + --local_rot bubble. 수정 Gregory cmdline.txt 의 새 인자(--gregory 뺌)와도 대조
BASE=$(grep -a -m1 '^\[BASE\] ' $O0/run.out | sed 's/^\[BASE\] //'); OV=$(grep -a -m1 '^\[OV\] ' $O0/run.out | sed 's/^\[OV\] //')
[ -n "$BASE" ] && [ -n "$OV" ] || fail "ficus_v0 run.out 에서 [BASE]/[OV] 를 못 읽었다"
LROT="--local_rot bubble"
O=$OB/run; mkdir -p $O
norm(){ tr -s ' ' <<<"$1" | sed 's/^ //; s/ $//'; }
mkargs(){ norm "--infer --data $DATA --hold_traj $T --out $1/ov --tag OVN $BASE $OV $2 --iters 0 --eval_t0 0 --eval_len $3 --tb none"; }
NA=$(mkargs $O "$LROT" $LEN)
NE="AF_INV_LOG=$O/inv_ov.jsonl AF_ROLL_DUMP=$O/roll_ov.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0"
GA=$(grep -m1 '^  새 인자:' $OGF/cmdline.txt | sed 's/^  새 인자: *//' | sed "s#$OGF/run#$O#g; s# --gregory##")
LA=$(mkargs $O "" $LEN)
{ echo "[명령줄] 기준 = lbours/ficus_v0 (커밋 110ce74) run.out 의 [BASE]·[OV]; 롤아웃 커밋 $C"
  echo "  새 인자: $NA"; echo "  새 환경: $NE"
  echo "  선형(같은 틀, --local_rot 없음): $LA"
  echo "  수정 Gregory 새 인자에서 --gregory 를 빼고 출력 디렉토리만 바꾼 것과 선형 틀 대조:"
  diff <(tr ' ' '\n' <<<"$(norm "$GA")") <(tr ' ' '\n' <<<"$LA") | sed 's/^/    /'
  [ "$(norm "$GA")" = "$LA" ] && echo "  -> 같다" || echo "  -> 다르다"
  echo "  선형 틀 대비 새 인자 차이:"; diff <(tr ' ' '\n' <<<"$LA") <(tr ' ' '\n' <<<"$NA") | sed 's/^/    /'
} | tee $OB/cmdline.txt
[ "$(norm "$GA")" = "$LA" ] || fail "선형 틀이 수정 Gregory 명령줄(--gregory 뺌)과 다르다 ($OB/cmdline.txt)"
[ "$(diff <(tr ' ' '\n' <<<"$LA") <(tr ' ' '\n' <<<"$NA") | grep -c '^[<>]')" -eq 2 ] || fail "새 인자에 --local_rot bubble 말고 다른 차이가 있다"
echo "AF_ROLL ... python -u exe/train_deform.py $NA" > $O/cmd.txt
free_gpus(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<500{print $1}'; }
wait_gpu(){ local g=""; while [ -z "$g" ]; do g=$(free_gpus | head -1); [ -z "$g" ] && { date +%s > $OB/wait_gpu; sleep 120; }; done; rm -f $OB/wait_gpu; echo $g; }
# 2) 3 프레임 통합 시험: 선형 | 회전판 θ 강제 0 (AF_LOCAL_ROT_ZERO=1) | 회전판. 빈 카드 나눠 동시에
S=$OB/smoke; mkdir -p $S
if [ ! -f $S/SMOKE_OK ]; then
  GL=($(free_gpus)); while [ ${#GL[@]} -lt 1 ]; do sleep 120; GL=($(free_gpus)); done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $S/gpu_at_start.txt
  i=0; PIDS=()
  for k in lin zero rot; do
    [ -f $S/$k/roll_ov.pt ] && continue
    mkdir -p $S/$k; G=${GL[$(( i % ${#GL[@]} ))]}; i=$((i+1))
    case $k in lin) X="";; zero) X="$LROT";; rot) X="$LROT";; esac
    Z=""; [ $k = zero ] && Z="AF_LOCAL_ROT_ZERO=1"
    A3=$(mkargs $S/$k "$X" 3)
    echo "AF_INV_LOG=$S/$k/inv_ov.jsonl AF_ROLL_DUMP=$S/$k/roll_ov.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 $Z CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py $A3" > $S/$k/cmd.txt
    echo "[시험 실행] $k GPU$G"
    env AF_INV_LOG=$S/$k/inv_ov.jsonl AF_ROLL_DUMP=$S/$k/roll_ov.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 $Z CUDA_VISIBLE_DEVICES=$G \
      python -u exe/train_deform.py $A3 < /dev/null > $S/$k/v_ov.log 2>&1 &
    PIDS+=($!)
    [ ${#GL[@]} -lt 3 ] && wait ${PIDS[-1]}      # 빈 카드가 모자라면 차례로
  done
  wait
  for k in lin zero rot; do [ -f $S/$k/roll_ov.pt ] || fail "3 프레임 시험 $k 실패 $(grep -a 'Traceback\|Error' $S/$k/v_ov.log | tail -2)"; done
  python - $S $O0/roll_ov.pt <<'PY' | tee $S/smoke.txt
import sys, json, torch
S, ref = sys.argv[1], sys.argv[2]
h = 0.056569
L = {k: torch.load(f"{S}/{k}/roll_ov.pt", map_location="cpu", weights_only=False) for k in ("lin", "zero", "rot")}
R0 = torch.load(ref, map_location="cpu", weights_only=False, mmap=True)
x0 = L["lin"]["x0"].float()
def dmax(a, b): return float((a.float() - b.float()).norm(dim=-1).max()) / h
for t in range(3):
    mv = float((L["lin"]["pred"][t].float() - x0).norm(dim=-1).mean()) / h
    print(f"[시험] 프레임 {t+1}: 선형 평균 이동 {mv:.4e} h | θ=0 회전판 vs 선형 최대 {dmax(L['zero']['pred'][t], L['lin']['pred'][t]):.3e} h | "
          f"선형(이 커밋) vs ficus_v0 덤프 최대 {dmax(L['lin']['pred'][t], R0['pred'][t]):.3e} h | 회전판 vs 선형 최대 {dmax(L['rot']['pred'][t], L['lin']['pred'][t]):.3e} h")
    print(f"        F: θ=0 vs 선형 최대 {float((L['zero']['F'][t] - L['lin']['F'][t]).abs().max()):.3e}, 회전판 vs 선형 최대 {float((L['rot']['F'][t] - L['lin']['F'][t]).abs().max()):.3e}")
for k in ("zero", "rot"):
    for ln in open(f"{S}/{k}/inv_ov.jsonl"):
        r = json.loads(ln); q = r.get("local_rot", {})
        print(f"[시험 {k}] 프레임 {r['frame']}: det F 최소 {r['det_min']:.4f}, det J_loc 최소 {q.get('detJloc_min', float('nan')):.5f}, 셀 밖 {q.get('out_of_cell')}, "
              f"|φ| 최대 {q.get('phi_max', float('nan')):.3e} 평균 {q.get('phi_mean', float('nan')):.3e}, |θ_p|/θmax 최대 {q.get('thp_over_max', float('nan')):.3f}, "
              f"‖K‖_F 최대 {q.get('K_fro_max', float('nan')):.3e}, θmax {q.get('theta_max_med', float('nan')):.4f}")
for k in ("lin", "zero", "rot"):
    for ln in open(f"{S}/{k}/v_ov.log", errors="replace"):
        if "[롤아웃] F_pg_v0" in ln: print(f"[시험 {k}] {ln.strip()[:160]}")
PY
  grep -q "회전판 vs 선형" $S/smoke.txt || fail "3 프레임 시험 비교 실패"
  touch $S/SMOKE_OK
fi
rclone copy $OB $R -q
# 3) 출력만 최적화 125 프레임 (덤프가 있으면 건너뜀)
if [ ! -f $O/roll_ov.pt ]; then
  G=$(wait_gpu); nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $OB/gpu_at_start.txt
  ts=$(date +%m%d_%H%M); for f in inv_ov.jsonl v_ov.log; do [ -f $O/$f ] && mv $O/$f $O/$f.partial_$ts; done
  echo "[실행] GPU$G $(date) 커밋 $C" | tee -a $OB/LOCALROT_LAUNCHED; cat $OB/gpu_at_start.txt; rclone copy $OB $R -q --max-depth 1
  env $NE CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py $NA < /dev/null > $O/v_ov.log 2>&1
  grep -a "롤아웃\] $T\|Traceback\|Error" $O/v_ov.log | tail -4 | cut -c1-300
  [ -f $O/roll_ov.pt ] || fail "롤아웃 실패 $(grep -a 'Traceback\|Error' $O/v_ov.log | tail -2)"
  echo "ov $(grep -a "롤아웃\] $T" $O/v_ov.log | tail -1 | grep -o '평균 [0-9.]*%')" > $O/err_ov.txt; cat $O/err_ov.txt
  rclone copy $OB $R -q
fi
G=$(wait_gpu)
[ -f $O/h5_ov/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_ov.pt --traj $GTD --out $O/h5_ov > $O/h5_ov.log 2>&1
[ -f $O/ov_ficus.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_ov --config $GTD/config.json --shape ficus --out $O/ov_ficus.mp4 > $O/r_ov.log 2>&1
[ -f $O/ov_ficus.mp4 ] || fail "렌더 실패 $(tail -3 $O/r_ov.log)"
PS=$O/psi; mkdir -p $PS
[ -f $PS/psi_hist.npz ] || CUDA_VISIBLE_DEVICES=$G python -u exe/psi_color.py --gt_h5 $TOPO --dump $O/roll_ov.pt --config $GTD/config.json --out $PS \
   --eps 1e-3 --q 1 99 --range -2.9999966621398926 4.933352713584899 --cmap turbo --stats_frames 1 40 120 > $PS/psi_color.log 2>&1
grep -q PSI_COLOR_DONE $PS/psi_color.log || fail "psi 실패 $(tail -3 $PS/psi_color.log)"
grep -a "변형\]" $PS/psi_color.log | tail -8
[ -f $PS/psi_ov_ficus.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_ov --config $GTD/config.json --shape ficus \
     --colors $PS/col_ov.npy --out $PS/psi_ov_ficus.mp4 > $PS/r_ov.log 2>&1
[ -f $PS/psi_ov_ficus.mp4 ] || fail "탄성 색 렌더 실패 $(tail -3 $PS/r_ov.log)"
rclone copy $OB $R -q
# 4) 분석 (master 커밋)
cd $WA || exit 1
AN=$OB/ana; mkdir -p $AN
# 4a) 진동: 무게중심·줄기 끝 (ficus_greg_fix.sh 와 같은 정의)
[ -f $AN/osc.json ] || python - $PT $AN/osc.json "선형=$O0/roll_ov.pt" "수정 Gregory=$OGF/run/roll_ov.pt" "회전판=$O/roll_ov.pt" <<'PY'
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
# 4b) 빠른 진동 분석 (/home/dkta/work/ficosc.py 와 같은 식: 주 진동 방향 평균 변위, 프레임당 이동, ψ 평균)
[ -f $AN/osc_quick.json ] || python -u exe/ficus_osc_quick.py --gt_dir $GTD --run "lin=선형=$O0/roll_ov.pt" \
    --run "gfix=수정 Gregory=$OGF/run/roll_ov.pt" --run "rot=회전판=$O/roll_ov.pt" --out $AN/osc_quick.json > $AN/osc_quick.log 2>&1
grep -q OSC_DONE $AN/osc_quick.log || fail "빠른 진동 분석 실패 $(tail -3 $AN/osc_quick.log)"
# 4c) 에너지 (정답 | 선형 | 수정 Gregory | 회전판) + 기존 값 검산
EG=$AN/energy; mkdir -p $EG
if ! grep -q FICUS_ENERGY_OK $EG/energy.log 2>/dev/null; then
  python -u exe/ficus_energy_cmp.py --gt_pt $PT --gt_psi $O0/psi/psi_gt.npy --run "선형=$O0" --run "수정 Gregory=$OGF/run" --run "회전판=$O" \
     --out_json $EG/energy.json --fig $EG/ficus_localrot_energy.png --title "ficus" > $EG/energy.log 2>&1; cat $EG/energy.log
  grep -q FICUS_ENERGY_OK $EG/energy.log || fail "에너지 실패"
  python - $EG/energy.json $O0/energy/energy.json <<'PY' | tee $EG/energy_check.txt
import sys, json, numpy as np
n, o = (json.load(open(p)) for p in sys.argv[1:])
for a, b in (("정답 (PG)", "_gt"), ("선형", "_ov")):
    for k in "KE":
        r = np.max(np.abs(np.array(n[a][k]) - np.array(o[k + b])) / np.abs(np.array(o[k + b])))
        print(f"[에너지 검산] {a} {k}: ficus_v0/energy/energy.json 대비 최대 상대차 {r:.2e}")
PY
fi
rclone copy $OB $R -q
# 4d) 노드 회전 벡터 분포 (덤프 노드 출력 4~6 채널) + 프레임별 det J_loc·셀 밖 (inv_ov.jsonl)
[ -f $AN/theta_stats.json ] || python - $AN/theta_stats.json $O/roll_ov.pt $O/inv_ov.jsonl <<'PY' | tee $AN/theta_stats.txt
import sys, json, torch
D = torch.load(sys.argv[2], map_location="cpu", weights_only=False, mmap=True)
inv = [json.loads(l) for l in open(sys.argv[3])]
tm = inv[0]["local_rot"]["theta_max_med"]
rows = []
for t, nd in enumerate(D["nodes"]):
    th = nd[1].float()[:, 3:6]; n = th.norm(dim=1); nz = n > 0
    q = n[nz] if bool(nz.any()) else torch.zeros(1)
    eff = tm * torch.tanh(q / tm)
    rows.append({"t": t, "n_nodes": int(n.numel()), "n_nonzero": int(nz.sum()), "med": float(q.median()), "p90": float(q.quantile(0.9)),
                 "p99": float(q.quantile(0.99)), "max": float(q.max()), "over_tm_med": float(q.median() / tm), "over_tm_p99": float(q.quantile(0.99) / tm),
                 "eff_med": float(eff.median()), "eff_p99": float(eff.quantile(0.99)), "frac_sat": float((q > 2 * tm).float().mean())})
L = [r["local_rot"] for r in inv]
out = {"theta_max": tm, "nodes": rows, "inv": inv}
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False)
print(f"[θ] θmax (셀 상한, 모든 셀 같음) {tm:.4f} rad; det J_loc 최소 (전 프레임) {min(q['detJloc_min'] for q in L):.5f}, 셀 밖 입자 합 {int(sum(q['out_of_cell'] for q in L))}, "
      f"|φ| 최대 {max(q['phi_max'] for q in L):.4e}, det F 최소 {min(r['det_min'] for r in inv):.4f}")
for t in (0, 1, 10, 30, 60, 124):
    if t < len(rows):
        r = rows[t]
        print(f"[θ] t={t:3d}: |θ| 중앙 {r['med']:.3e} 90% {r['p90']:.3e} 99% {r['p99']:.3e} 최대 {r['max']:.3e} (θmax 배수 중앙 {r['over_tm_med']:.2f} 99% {r['over_tm_p99']:.2f}); "
              f"상한 뒤 |θ̃| 중앙 {r['eff_med']:.3e} 99% {r['eff_p99']:.3e}; 포화(|θ|>2θmax) {r['frac_sat']:.3f}; 0 아닌 노드 {r['n_nonzero']}/{r['n_nodes']}")
PY
[ -f $AN/theta_stats.json ] || fail "θ 분포 실패"
# 4e) 셀별 잠긴 에너지 (정답·선형·수정 Gregory 는 ficus_v0_greg_fix/ana/locked 결과 복사 재사용, 원본 손대지 않음)
LK=$AN/locked; mkdir -p $LK; LO=$OGF/ana/locked
if [ ! -f $LK/LOCKED_LOCALROT_DONE ]; then
  G=$(wait_gpu)
  for f in cells_gt.npz dens_gt.npy cells_lin.npz dens_lin.npy cells_gfix.npz dens_gfix.npy; do [ -f $LK/$f ] || cp $LO/$f $LK/$f; done
  [ -f $LK/summary.json ] || python -c "import json;s=json.load(open('$LO/summary.json'));json.dump({k:s[k] for k in ('gt','lin','gfix')},open('$LK/summary.json','w'),ensure_ascii=False,indent=0)"
  CUDA_VISIBLE_DEVICES=$G python -u exe/cell_locked_energy.py --config $GTD/config.json --vol_pt $PT --node_h 0.056569 --out $LK \
    --traj "gt:정답 (PG):#222222:h5:$TOPO" --traj "lin:선형:#d9480f:dump:$O0/roll_ov.pt" \
    --traj "gfix:수정 Gregory:#1c7ed6:dump:$OGF/run/roll_ov.pt" --traj "rot:회전판:#2b8a3e:dump:$O/roll_ov.pt" \
    --energy_json $EG/energy.json --energy_keys "gt=정답 (PG),lin=선형,gfix=수정 Gregory,rot=회전판" \
    --fig $LK/ficus_localrot_locked.png --table $LK/ficus_localrot_locked.md < /dev/null > $LK/run.log 2>&1
  grep -q LOCKED_CALC_DONE $LK/run.log || fail "잠긴 에너지 실패 $(grep -av 'it/s\|s/it' $LK/run.log | tail -3)"
  touch $LK/LOCKED_LOCALROT_DONE
fi
grep -a "재사용\|검산\|수렴" $LK/run.log | head -12
kill $SYNC 2>/dev/null; rclone copy $OB $R -q
touch $OB/LOCALROT_CLUSTER_DONE; rclone copy $OB $R -q; echo "LOCALROT_CLUSTER_DONE $(date)"
