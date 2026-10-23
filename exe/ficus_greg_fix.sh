#!/bin/bash
# ficus_v0_greg 다시 돌리기: 로컬 변형 버그 수정판 (사용자 10-23 "그거 버그 수정해서 제대로 다시 돌려").
# 이전 실행 /home/dkta/work/ficus_v0_greg.sh (커밋 c95f79c) 와 명령줄·학습 코드가 같고, 바뀐 것은 lib/anchorflow/skin.py
# (λ' = λ + W(ρ) − W(ρ_ref), x' = Σλ'(X+U)) 뿐이다 -- 롤아웃은 c95f79c 에 고친 skin.py 만 얹은 커밋(AF_RUN_COMMIT)에서,
# 분석(에너지·잠긴 에너지·분해·진동)은 master 커밋(AF_ANA_COMMIT)에서.
# 출력 lbours/ficus_v0_greg_fix (처음부터 R2 통째 동기화, 제외 없음). 멱등: 다시 띄우면 끝난 단계는 건너뛴다 (이어 돌리기 = 단계 단위).
# 표식: GREGFIX_LAUNCHED, GREGFIX_CLUSTER_DONE, GREGFIX_FAIL
# 사용: AF_RUN_COMMIT=... AF_ANA_COMMIT=... AF_DETACH=1 bash exe/job_run.sh /home/dkta/work/ficus_greg_fix.sh <로그>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
C=${AF_RUN_COMMIT:?롤아웃 커밋}; CA=${AF_ANA_COMMIT:?분석 커밋}
WT=$W/wt/gregfix_$C; WA=$W/wt/gregfix_ana_$CA
O0=$W/lbours/ficus_v0; OG=$W/lbours/ficus_v0_greg; TOPO=$W/topo/pg_ficus_elastic_v0
OB=$W/lbours/ficus_v0_greg_fix; R=r2:storage/result/anchorflow/lbours/ficus_v0_greg_fix
T=F_pg_v0; DATA=$O0/data/$T; PT=$DATA/$T.pt; GTD=$O0/gt; LEN=125; mkdir -p $OB
fail(){ echo "[오류] $1"; echo "$(date '+%m-%d %H:%M') $1" >> $OB/GREGFIX_FAIL; rclone copy $OB $R -q; exit 1; }
for p in "$WT $C" "$WA $CA"; do set -- $p
  [ -d $1 ] || (cd $W/anchorflow && git fetch -q origin && git worktree add -q --detach $1 $2) || fail "worktree $2"; done
if [ -f $OB/driver.hb ] && [ $(( $(date +%s) - $(cat $OB/driver.hb) )) -lt 300 ]; then echo "[중단] 다른 드라이버가 돌고 있다 $(date)"; exit 0; fi
echo $$ > $OB/driver.pid
( while true; do date +%s > $OB/driver.hb; sleep 120; done ) & HB=$!
( while true; do rclone copy $OB $R -q; sleep 300; done ) & SYNC=$!
trap 'kill $SYNC $HB 2>/dev/null' EXIT
cd $WT || exit 1
echo "[시작] $(date) 롤아웃 커밋 $(git rev-parse --short HEAD) 분석 커밋 $CA 노드 $(hostname) 잡 ${PBS_JOBID:-?}"
git diff --stat c95f79c HEAD | tee $OB/code_diff_vs_c95f79c.txt
# 0) 단위 시험 (CPU)
if [ ! -f $OB/unit_$C.ok ]; then
  CUDA_VISIBLE_DEVICES= python -u exe/test_greg_local.py --pt $PT > $OB/unit_$C.log 2>&1; cat $OB/unit_$C.log
  grep -q "\[전체\] 통과" $OB/unit_$C.log || fail "단위 시험 실패 $OB/unit_$C.log"
  touch $OB/unit_$C.ok
fi
# 1) 명령줄: ficus_v0_greg.sh 원문 풀기 vs 새 명령 (출력 디렉토리만 같게 풀어 비교)
COMMON="--no_mat --arch sgnn --gnn_layers 8 --hidden 128 --obj pts --net_inertial"
FLOOR="--bc_level node --bc_mode project"
REST="--inv_barrier 0 --node_h 0.056569 --dt_cond --dt_scale --v0_traj --det_eps 0.1 --det_w 0 --det_keep_elastic --lr 3e-4 --batch 8 --det_every 200 --pool_fill $W/lbpool/fill --pool_cfg $W/lbpool/cfg --pool_combos lego_jelly --pool_drop 0.0,0.02,0.0,1.5 --pool_frames 80 --phase2 --phys_w 1.0 --save_every 100000"
PGF="--cfg_bc --pg_region_density --pg_vdamp"
BASE="$COMMON $FLOOR $REST $PGF"; GREG="--gregory"
OV="--out_var --ov_init predict --ov_roll 500 --ov_opt lbfgs --ov_vars out --ov_obj ip"
O=$OB/run; mkdir -p $O
NA="--infer --data $DATA --hold_traj $T --out $O/ov --tag OVN $BASE $OV $GREG --iters 0 --eval_t0 0 --eval_len $LEN --tb none"
NE="AF_INV_LOG=$O/inv_ov.jsonl AF_ROLL_DUMP=$O/roll_ov.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0"
norm(){ tr -s ' ' <<<"$1" | sed 's/^ //; s/ $//'; }
S0=$W/ficus_v0_greg.sh
DEFS=$(grep -E '^(COMMON|FLOOR|REST|PGF|BASE|GREG|OV)=' $S0)
PL=$(grep -m1 'python -u exe/train_deform.py' $S0 | sed 's/ > \$O\/v_ov.log.*//; s/^.*python -u exe\/train_deform.py //')
EL=$(grep -m1 'AF_INV_LOG=' $S0 | sed 's/.*|| //; s/ CUDA_VISIBLE_DEVICES.*//; s/^ *//')
OL=$(W=$W bash -c "$DEFS; DATA=$DATA; T=$T; O=$O; LEN=$LEN; echo \"$PL\"; echo \"$EL\"")
OA=$(norm "$(sed -n 1p <<<"$OL")"); OE=$(norm "$(sed -n 2p <<<"$OL")"); NA=$(norm "$NA"); NE=$(norm "$NE")
{ echo "[명령줄 대조] 기준 $S0 (커밋 c95f79c) vs 새 (커밋 $C), 출력 디렉토리는 같게 풀어 비교"
  echo "  기준 인자: $OA"; echo "  새 인자:   $NA"; echo "  기준 환경: $OE"; echo "  새 환경:   $NE"
  echo "  인자 차이:"; diff <(tr ' ' '\n' <<<"$OA") <(tr ' ' '\n' <<<"$NA") | sed 's/^/    /'
  if [ "$OA" = "$NA" ] && [ "$OE" = "$NE" ]; then echo "  -> 같다 (커밋만 다름)"; else echo "  -> 다르다"; fi
} | tee $OB/cmdline.txt
[ "$OA" = "$NA" ] && [ "$OE" = "$NE" ] || fail "명령줄이 기준과 다르다 ($OB/cmdline.txt)"
echo "AF_ROLL ... python -u exe/train_deform.py $NA" > $O/cmd.txt
free_gpu(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<500{print $1}' | head -1; }
wait_gpu(){ local g=""; while [ -z "$g" ]; do g=$(free_gpu); [ -z "$g" ] && { date +%s > $OB/wait_gpu; sleep 120; }; done; rm -f $OB/wait_gpu; echo $g; }
# 2) 출력만 최적화 125 프레임 (덤프가 있으면 건너뜀)
if [ ! -f $O/roll_ov.pt ]; then
  G=$(wait_gpu); nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $OB/gpu_at_start.txt
  ts=$(date +%m%d_%H%M); for f in inv_ov.jsonl v_ov.log; do [ -f $O/$f ] && mv $O/$f $O/$f.partial_$ts; done
  echo "[실행] GPU$G $(date) 커밋 $C" | tee -a $OB/GREGFIX_LAUNCHED; cat $OB/gpu_at_start.txt; rclone copy $OB $R -q --max-depth 1
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
rclone copy $OB $R -q
# 3) 분석 (master 커밋)
cd $WA || exit 1
AN=$OB/ana; mkdir -p $AN
# 3a) 진동: 무게중심·줄기 끝 (ficus_v0_dt.sh 와 같은 정의), 네 궤적
[ -f $AN/osc.json ] || python - $PT $AN/osc.json "선형=$O0/roll_ov.pt" "이전 Gregory=$OG/roll_ov.pt" "수정 Gregory=$O/roll_ov.pt" <<'PY'
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
# 3b) 에너지 (정답 | 선형 | 이전 Gregory | 수정 Gregory) + 기존 값 검산
EG=$AN/energy; mkdir -p $EG
if ! grep -q FICUS_ENERGY_OK $EG/energy.log 2>/dev/null; then
  python -u exe/ficus_energy_cmp.py --gt_pt $PT --gt_psi $O0/psi/psi_gt.npy --run "선형=$O0" --run "이전 Gregory=$OG" --run "수정 Gregory=$O" \
     --out_json $EG/energy.json --fig $EG/ficus_greg_fix_energy.png --title "ficus" > $EG/energy.log 2>&1; cat $EG/energy.log
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
# 3c) 노드 출력 분포 (변위 |U|/h, 반경 ρ_raw·ρ/h, 로컬 단계 입자 이동 |Δx_loc|/h) -- 이전·수정 Gregory
[ -f $AN/nodes_stats.json ] || CUDA_VISIBLE_DEVICES=$G python - $AN/nodes_stats.json "이전 Gregory=$OG/roll_ov.pt" "수정 Gregory=$O/roll_ov.pt" <<'PY' | tee $AN/nodes_stats.txt
import sys, json, torch
h = 0.056569; out = {}
for q in sys.argv[2:]:
    k, p = q.split("=", 1)
    D = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    rows = []
    for t, nd in enumerate(D["nodes"]):
        dp = nd[1].float(); U, Rr = dp[:, :3], dp[:, 3]
        un = U.norm(dim=1) / h; rho = 0.05 + 0.95 * torch.sigmoid(Rr)
        rows.append({"t": t, "U_med_h": float(un.median()), "U_p99_h": float(un.quantile(0.99)), "U_max_h": float(un.max()),
                     "rho_raw_med": float(Rr.median()), "rho_raw_p1": float(Rr.quantile(0.01)), "rho_raw_p99": float(Rr.quantile(0.99)),
                     "rho_raw_absmax": float(Rr.abs().max()), "rho_h_min": float(rho.min()), "rho_h_max": float(rho.max()),
                     "frac_rho_raw_gt_0p1": float((Rr.abs() > 0.1).float().mean())})
    out[k] = rows
    for t in (0, 1, 10, 30, 60, 124):
        if t < len(rows):
            r = rows[t]
            print(f"[노드] {k} t={t:3d}: |U|/h 중앙 {r['U_med_h']:.3e} 99% {r['U_p99_h']:.3e} 최대 {r['U_max_h']:.3e} | ρ_raw 1% {r['rho_raw_p1']:+.3f} "
                  f"중앙 {r['rho_raw_med']:+.3f} 99% {r['rho_raw_p99']:+.3f} |max| {r['rho_raw_absmax']:.3f} | ρ/h [{r['rho_h_min']:.3f}, {r['rho_h_max']:.3f}] | |ρ_raw|>0.1 비율 {r['frac_rho_raw_gt_0p1']:.3f}")
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False)
PY
# 3d) 스텝별 탄성 에너지 변화의 로컬/글로벌 분해 (고친 스키닝; CPU)
SP=$AN/split_el; mkdir -p $SP
if ! grep -q GREG_SPLIT_DONE $SP/run.out 2>/dev/null; then
  ts=$(date +%m%d_%H%M); for f in run.out greg_split_el.json; do [ -f $SP/$f ] && mv $SP/$f $SP/$f.partial_$ts; done
  # 시험: 노드 집합이 바뀐 스텝(78~80) 먼저
  python -u exe/greg_split_el.py --skin fix --dump $O/roll_ov.pt --data $PT --t0 78 --t1 81 --out $SP/check_t78_80.json --threads 8 > $SP/check_t78_80.out 2>&1
  grep -q GREG_SPLIT_DONE $SP/check_t78_80.out || fail "분해 시험(78~80) 실패 $(tail -2 $SP/check_t78_80.out)"
  grep "^t=" $SP/check_t78_80.out | cut -c1-200
fi
grep -q GREG_SPLIT_DONE $SP/run.out 2>/dev/null || python -u exe/greg_split_el.py --skin fix --dump $O/roll_ov.pt --data $PT --t1 $LEN --out $SP/greg_split_el.json --threads 8 > $SP/run.out 2>&1
grep -q GREG_SPLIT_DONE $SP/run.out || fail "분해 실패 $(tail -3 $SP/run.out)"
tail -3 $SP/run.out | cut -c1-400
rclone copy $OB $R -q
# 3e) 셀별 잠긴 에너지 (정답·선형은 lbours/ficus_locked 결과 복사 재사용, 원본 손대지 않음)
LK=$AN/locked; mkdir -p $LK; LO=$W/lbours/ficus_locked
if [ ! -f $LK/LOCKED_GREGFIX_DONE ]; then
  for f in cells_gt.npz dens_gt.npy cells_lin.npz dens_lin.npy; do [ -f $LK/$f ] || cp $LO/$f $LK/$f; done
  [ -f $LK/summary.json ] || python -c "import json;s=json.load(open('$LO/summary.json'));json.dump({k:s[k] for k in ('gt','lin')},open('$LK/summary.json','w'),ensure_ascii=False,indent=0)"
  CUDA_VISIBLE_DEVICES=$G python -u exe/cell_locked_energy.py --config $GTD/config.json --vol_pt $PT --node_h 0.056569 --out $LK \
    --traj "gt:정답 (PG):#222222:h5:$TOPO" --traj "lin:선형:#d9480f:dump:$O0/roll_ov.pt" \
    --traj "gold:이전 Gregory:#868e96:dump:$OG/roll_ov.pt" --traj "gfix:수정 Gregory:#1c7ed6:dump:$O/roll_ov.pt" \
    --energy_json $EG/energy.json --energy_keys "gt=정답 (PG),lin=선형,gold=이전 Gregory,gfix=수정 Gregory" \
    --fig $LK/ficus_greg_fix_locked.png --table $LK/ficus_greg_fix_locked.md < /dev/null > $LK/run.log 2>&1
  grep -q LOCKED_CALC_DONE $LK/run.log || fail "잠긴 에너지 실패 $(grep -av 'it/s\|s/it' $LK/run.log | tail -3)"
  touch $LK/LOCKED_GREGFIX_DONE
fi
grep -a "재사용\|검산\|수렴" $LK/run.log | head -12
kill $SYNC 2>/dev/null; rclone copy $OB $R -q
touch $OB/GREGFIX_CLUSTER_DONE; rclone copy $OB $R -q; echo "GREGFIX_CLUSTER_DONE $(date)"
