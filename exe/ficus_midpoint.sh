#!/bin/bash
# ficus_v0 출력만 최적화: 적분기만 암시적 중점법(--integrator midpoint)으로 바꾼 비교 (사용자 10-21).
# 기준 = lbours/ficus_v0 (후퇴 오일러, 110ce74, 1.788%) 덤프 재사용 -- 재실행 없음.
# 명령줄은 /home/dkta/work/ficus_v0.sh 원문을 풀어 자동 대조 (차이는 --integrator midpoint 와 출력 디렉토리뿐이어야 한다).
# 출력 lbours/ficus_midpoint (처음부터 R2 통째 동기화, 제외 없음). 멱등: 다시 띄우면 끝난 단계는 건너뛴다.
# 표식: MIDPOINT_LAUNCHED(롤아웃 시작), MIDPOINT_CLUSTER_DONE, MIDPOINT_FAIL
# 사용: AF_DETACH=1 bash exe/job_run.sh /home/dkta/work/ficus_midpoint.sh <로그>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
C=${AF_MID_COMMIT:?커밋}; WT=$W/wt/midpoint_$C
O0=$W/lbours/ficus_v0; TOPO=$W/topo/pg_ficus_elastic_v0; OB=$W/lbours/ficus_midpoint; R=r2:storage/result/anchorflow/lbours/ficus_midpoint
T=F_pg_v0; DATA=$O0/data/$T; GTD=$O0/gt; LEN=125; mkdir -p $OB
fail(){ echo "[오류] $1"; echo "$(date '+%m-%d %H:%M') $1" >> $OB/MIDPOINT_FAIL; rclone copy $OB $R -q; exit 1; }
[ -d $WT ] || (cd $W/anchorflow && git fetch -q origin && git worktree add -q --detach $WT $C) || fail "worktree $C"
cd $WT || exit 1
if [ -f $OB/driver.hb ] && [ $(( $(date +%s) - $(cat $OB/driver.hb) )) -lt 300 ]; then echo "[중단] 다른 드라이버가 돌고 있다 (하트비트 5 분 안) $(date)"; exit 0; fi
echo $$ > $OB/driver.pid
( while true; do date +%s > $OB/driver.hb; sleep 120; done ) & HB=$!
( while true; do rclone copy $OB $R -q; sleep 300; done ) & SYNC=$!
trap 'kill $SYNC $HB 2>/dev/null' EXIT
echo "[시작] $(date) 커밋 $(git rev-parse --short HEAD) 노드 $(hostname) 잡 ${PBS_JOBID:-?}"
# 0) 단위 시험 (CPU)
if [ ! -f $OB/unit_$C.ok ]; then
  CUDA_VISIBLE_DEVICES= python -u exe/test_midpoint.py > $OB/unit_$C.log 2>&1; cat $OB/unit_$C.log
  grep -q "MIDPOINT_TEST PASS" $OB/unit_$C.log || fail "단위 시험 실패 $OB/unit_$C.log"
  touch $OB/unit_$C.ok; rclone copy $OB $R -q --max-depth 1
fi
# 1) 명령줄: ficus_v0.sh 원문 풀기 vs 새 명령
COMMON="--no_mat --arch sgnn --gnn_layers 8 --hidden 128 --obj pts --net_inertial"
FLOOR="--bc_level node --bc_mode project"
REST="--inv_barrier 0 --node_h 0.056569 --dt_cond --dt_scale --v0_traj --det_eps 0.1 --det_w 0 --det_keep_elastic --lr 3e-4 --batch 8 --det_every 200 --pool_fill $W/lbpool/fill --pool_cfg $W/lbpool/cfg --pool_combos lego_jelly --pool_drop 0.0,0.02,0.0,1.5 --pool_frames 80 --phase2 --phys_w 1.0 --save_every 100000"
PGF="--cfg_bc --pg_region_density --pg_vdamp"
BASE="$COMMON $FLOOR $REST $PGF"
OV="--out_var --ov_init predict --ov_roll 500 --ov_opt lbfgs --ov_vars out --ov_obj ip"
MID="--integrator midpoint"
args_of(){ # 출력디렉토리 프레임수 추가플래그
  echo "--infer --data $DATA --hold_traj $T --out $1/ov --tag OVN $BASE $OV $3 --iters 0 --eval_t0 0 --eval_len $2 --tb none"; }
env_of(){ echo "AF_INV_LOG=$1/inv_ov.jsonl AF_ROLL_DUMP=$1/roll_ov.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0"; }
norm(){ tr -s ' ' <<<"$1" | sed 's/^ //; s/ $//'; }
S0=$W/ficus_v0.sh
DEFS=$(grep -E '^(COMMON|FLOOR|REST|PGF|BASE|OV)=' $S0)
PL=$(grep -m1 'python -u exe/train_deform.py' $S0 | sed 's/ < \/dev\/null.*//; s/ > \$O\/v_ov.log.*//; s/^.*python -u exe\/train_deform.py //')
EL=$(grep -m1 'AF_INV_LOG=' $S0 | sed 's/.*|| //; s/ CUDA_VISIBLE_DEVICES.*//; s/^ *//')
cmpl(){ # 이름 출력디렉토리 프레임수 추가플래그
  local OL=$(W=$W bash -c "$DEFS; DATA=$DATA; T=$T; O=$2; LEN=$3; echo \"$PL\"; echo \"$EL\"")
  local OA=$(norm "$(sed -n 1p <<<"$OL")") OE=$(norm "$(sed -n 2p <<<"$OL")") NA=$(norm "$(args_of $2 $3 "$4")") NE=$(norm "$(env_of $2)")
  local NAX=$(norm "$(sed "s/ $4 / /" <<<" $NA ")")
  { echo "[명령줄 대조 $1] 기준 $S0 (커밋 110ce74) vs 새 (커밋 $C), 출력 디렉토리는 같게 풀어 비교"
    echo "  기준 인자: $OA"; echo "  새 인자:   $NA"; echo "  기준 환경: $OE"; echo "  새 환경:   $NE"
    echo "  인자 차이 (기준 -> 새):"; diff <(tr ' ' '\n' <<<"$OA") <(tr ' ' '\n' <<<"$NA") | sed 's/^/    /'
    if [ "$OA" = "$NAX" ] && [ "$OE" = "$NE" ]; then echo "  -> 같다 (추가 플래그 '${4:-없음}' 와 커밋만 다름)"; else echo "  -> 다르다"; fi
  } | tee $OB/cmdline_$1.txt
  [ "$OA" = "$NAX" ] && [ "$OE" = "$NE" ]; }
mkdir -p $OB/mid $OB/be_check
cmpl mid $OB/mid $LEN "$MID" || fail "명령줄 mid 가 기준과 다르다 ($OB/cmdline_mid.txt)"
cmpl be_check $OB/be_check 5 "" || fail "명령줄 be_check 가 기준과 다르다"
free_gpu(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<1024{print $1}' | head -1; }
wait_gpu(){ local g=""; while [ -z "$g" ]; do g=$(free_gpu); [ -z "$g" ] && { date +%s > $OB/wait_gpu; sleep 60; }; done; rm -f $OB/wait_gpu; echo $g; }
G=$(wait_gpu); nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $OB/gpu_at_start.txt
echo "[GPU] $G $(date)" | tee -a $OB/GPU; cat $OB/gpu_at_start.txt
run(){ # 이름 프레임수 추가플래그
  local O=$OB/$1 ts=$(date +%m%d_%H%M)
  [ -f $O/roll_ov.pt ] && return 0
  for f in inv_ov.jsonl v_ov.log; do [ -f $O/$f ] && mv $O/$f $O/$f.partial_$ts; done
  echo "[실행] $1 GPU$G $(date) 커밋 $C" | tee -a $OB/MIDPOINT_LAUNCHED
  rclone copy $OB $R -q --max-depth 1
  env $(env_of $O) CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py $(args_of $O $2 "$3") < /dev/null > $O/v_ov.log 2>&1
  grep -a "적분기\]\|롤아웃\] $T\|Traceback\|Error" $O/v_ov.log | tail -4 | cut -c1-300
  [ -f $O/roll_ov.pt ] || return 1
  echo "$1 $(grep -a "롤아웃\] $T" $O/v_ov.log | tail -1 | grep -o '평균 [0-9.]*%')" > $O/err_ov.txt; cat $O/err_ov.txt; }
# 2) 기본 경로 회귀 확인 (새 커밋, --integrator 없음 = 후퇴 오일러, 5 프레임) -- 기준 덤프와 같은지만 본다 (결과로 쓰지 않음)
run be_check 5 "" || fail "be_check 실패 $(tail -3 $OB/be_check/v_ov.log)"
python - $OB/be_check/roll_ov.pt $O0/roll_ov.pt $DATA/$T.pt <<'PY' | tee $OB/be_check/be_check.txt
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False); b = torch.load(sys.argv[2], map_location="cpu", weights_only=False, mmap=True)
g = torch.load(sys.argv[3], map_location="cpu", weights_only=False, mmap=True)
x0 = g["x"][0].float(); ext = float((x0.max(0).values - x0.min(0).values).norm())
for i in range(a["pred"].shape[0]):
    dx = float((a["pred"][i].float() - b["pred"][i].float()).norm(dim=-1).max()); dv = float((a["v"][i].float() - b["v"][i].float()).norm(dim=-1).max())
    mv = float((b["pred"][i].float() - x0).norm(dim=-1).mean())
    print(f"[회귀] 프레임 {i + 1}: 새 커밋 기본 경로 vs 기준 덤프 위치 최대차 {dx:.3e} (대각선 {ext:.3f} 대비 {dx / ext:.2e}, 기준 평균 이동 {mv:.3e}), 속도 최대차 {dv:.3e}")
PY
# 3) 중점법 125 프레임
run mid $LEN "$MID" || fail "mid 롤아웃 실패 $(grep -a 'Traceback\|Error' $OB/mid/v_ov.log | tail -2)"
O=$OB/mid
python - $O/inv_ov.jsonl <<'PY' | tee $O/inv_summary.txt
import json, sys
R = [json.loads(l) for l in open(sys.argv[1])]
mx = lambda k: max(r.get(k, -1) for r in R); mn = lambda k: min(r.get(k, float("nan")) for r in R)
fr = [r["frame"] for r in R if r.get("ext_inv_particles", 0) > 0]
print(f"[뒤집힘] 프레임 {len(R)}; 중점 사상 det<=0 입자 최대 {mx('mid_inv_particles')}, 셀 최대 {mx('mid_inv_tets')}, det 최소 {mn('mid_det_min'):.4e}; "
      f"외삽 사상(I+2∇u) det<=0 입자 최대 {mx('ext_inv_particles')}, 셀 최대 {mx('ext_inv_tets')} (점유 셀 {R[-1].get('tets')}), det 최소 {mn('ext_det_min'):.4e}, "
      f"det<=0 있는 프레임 {len(fr)} {fr[:12]}; 누적 det F<=0 입자 최대 {mx('inv_particles_cum')}")
print(f"[경계조건] 구속 입자 {R[0].get('bc_pinned')} (최대 {mx('bc_pinned')}), 중점 변위 최대 {mx('bc_mid_disp_max'):.3e}, 끝 변위 최대 {mx('bc_end_disp_max'):.3e}, 끝 속도 최대 {mx('bc_end_v_max'):.3e}")
PY
GOOD=1
[ -f $O/h5_ov/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_ov.pt --traj $GTD --out $O/h5_ov > $O/h5_ov.log 2>&1
[ -f $O/ov_ficus.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_ov --config $GTD/config.json --shape ficus --out $O/ov_ficus.mp4 > $O/r_ov.log 2>&1
[ -f $O/ov_ficus.mp4 ] || fail "렌더 실패 $(tail -3 $O/r_ov.log)"
PS=$O/psi; mkdir -p $PS
[ -f $PS/psi_hist.npz ] || CUDA_VISIBLE_DEVICES=$G python -u exe/psi_color.py --gt_h5 $TOPO --dump $O/roll_ov.pt --config $GTD/config.json --out $PS \
   --eps 1e-3 --q 1 99 --range -2.9999966621398926 4.933352713584899 --cmap turbo --stats_frames 1 40 120 > $PS/psi_color.log 2>&1
grep -q PSI_COLOR_DONE $PS/psi_color.log || fail "psi 실패 $(tail -3 $PS/psi_color.log)"
grep -a "변형\]" $PS/psi_color.log | tail -8
rclone copy $OB $R -q
# 4) 에너지 곡선 (정답 | 후퇴 오일러(기준) | 중점법) + 기준 값 검산 (lbours/ficus_v0/energy/energy.json)
EG=$OB/energy; mkdir -p $EG
python -u exe/ficus_energy_cmp.py --gt_pt $DATA/$T.pt --gt_psi $O0/psi/psi_gt.npy --run "후퇴 오일러 (기준)=$O0" --run "중점법=$O" \
   --out_json $EG/energy.json --fig $EG/ficus_midpoint_energy.png --title "ficus" > $EG/energy.log 2>&1; cat $EG/energy.log
grep -q FICUS_ENERGY_OK $EG/energy.log || fail "에너지 실패"
python - $EG/energy.json $O0/energy/energy.json <<'PY' | tee $EG/energy_check.txt
import sys, json, numpy as np
n, o = (json.load(open(p)) for p in sys.argv[1:])
for a, b in (("정답 (PG)", "_gt"), ("후퇴 오일러 (기준)", "_ov")):
    for k in "KE":
        r = np.max(np.abs(np.array(n[a][k]) - np.array(o[k + b])) / np.abs(np.array(o[k + b])))
        print(f"[에너지 검산] {a} {k}: 기존 energy.json 대비 최대 상대차 {r:.2e}")
PY
rclone copyto $EG/ficus_midpoint_energy.png r2:storage/result/anchorflow/ovnet/ficus_midpoint_energy.png -q
rclone copy $OB $R -q
# 5) 셀별 잠긴 에너지 (cell_locked_energy.py; 정답·후퇴 오일러는 lbours/ficus_locked 결과를 복사해 재사용, 원본 손대지 않음)
LK=$OB/locked; mkdir -p $LK; LO=$W/lbours/ficus_locked
if [ ! -f $LK/LOCKED_MID_DONE ]; then
  for f in cells_gt.npz dens_gt.npy cells_lin.npz dens_lin.npy; do [ -f $LK/$f ] || cp $LO/$f $LK/$f; done
  [ -f $LK/summary.json ] || python -c "import json;s=json.load(open('$LO/summary.json'));json.dump({k:s[k] for k in ('gt','lin')},open('$LK/summary.json','w'),ensure_ascii=False,indent=0)"
  CUDA_VISIBLE_DEVICES=$G python -u exe/cell_locked_energy.py --config $GTD/config.json --vol_pt $DATA/$T.pt --node_h 0.056569 --out $LK \
    --traj "gt:정답 (PG):#222222:h5:$TOPO" --traj "lin:후퇴 오일러 (기준):#d9480f:dump:$O0/roll_ov.pt" --traj "mid:중점법:#1c7ed6:dump:$O/roll_ov.pt" \
    --energy_json $EG/energy.json --energy_keys "gt=정답 (PG),lin=후퇴 오일러 (기준),mid=중점법" \
    --fig $LK/ficus_midpoint_locked.png --table $LK/ficus_midpoint_locked.md < /dev/null > $LK/run.log 2>&1
  grep -q LOCKED_CALC_DONE $LK/run.log || fail "잠긴 에너지 실패 $(grep -av 'it/s\|s/it' $LK/run.log | tail -3)"
  touch $LK/LOCKED_MID_DONE
fi
grep -a "재사용\|검산\|수렴" $LK/run.log | head -12
kill $SYNC 2>/dev/null; rclone copy $OB $R -q
touch $OB/MIDPOINT_CLUSTER_DONE; rclone copy $OB $R -q; echo "MIDPOINT_CLUSTER_DONE $(date)"
