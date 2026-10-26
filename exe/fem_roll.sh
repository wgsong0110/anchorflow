#!/bin/bash
# 모델 학습 대신 격자 업데이트를 FEM 으로 (사용자 10-26 "lego랑 ficus에 대해서 모델 학습 시키는 대신에 격자 업데이트를 FEM으로 하는걸로 해서도 실행해줘봐").
#   매 프레임 Kuhn 사면체 격자 노드 변위를 망 없이 같은 증분 포텐셜(관성 + 입자 구적 탄성 + 바닥·경계, 그 씬 기존 실행과 같은 플래그)의
#   뉴턴-CG 직접 풀이로 **수렴까지** 구한다: train_deform --out_var --ov_vars out --ov_obj newton + AF_FEM_TOL (수렴 모드, 같은 커밋에서 추가).
#   기존 출력만 최적화(L-BFGS 500 고정) 실행과 명령줄 차이는 --ov_obj ip -> newton 과 --net_inertial 제거(10-22 지시)뿐 -- 자동 대조 (OVN.json args).
#   걸음: 감쇠 없는 뉴턴 (AF_OV_NEWTON_DAMP=0), CG 최대 AF_OV_NEWTON_CG, 아르미호 되추적 선탐색, 수렴 = |p|∞/(h·EXT) < AF_FEM_TOL (속도 단위, dt 비례),
#   최대 AF_OV_NEWTON_STEPS 걸음, E 가 3 걸음 연속 상대 1e-7 도 안 줄면 정체(float32 잡음 바닥)로 멈춤. 허용치: FEM_TOL (lego 1e-2 = IPC 기본, float32 바닥 ~4e-3 위; ficus 1e-3). 프레임마다 [FEM 뉴턴] 줄 + fem.jsonl (반복수·잔차·선탐색·시간), tqdm "FEM 롤아웃".
#   정답 궤적 = 그 형상-물성의 기존 평가 궤적 (lego L000 = one/L000 80 프레임, ficus = lbours/ficus_v0 F_pg_v0 125 프레임), 같은 초기 조건.
#   오차 = 롤아웃 [롤아웃] 줄의 평균 % (기존 실행과 같은 지표, 고정 상수 EXT 정규화) -> err_fem.txt, 기존 실행 값과 나란히 err_cmp.txt.
#   영상 = 3DGS 알파블렌딩 (lb_render) 정답 | FEM 나란히, 프레임 번호·한국어 라벨 -> cmp_<씬>.mp4.
# 사용: bash exe/fem_roll.sh <lego|ficus> <GPU> <test|full>   (잡 안에서만: AF_DETACH=1 bash exe/job_run.sh <부트 스크립트> <로그>)
#   test = 1 프레임 ($U/test_<커밋>), full = 전체. 출력 lbours/fem_<씬> (처음부터 R2 통째 동기화, 제외 없음).
#   이어 돌리기: 10 프레임마다 조각(AF_ROLL_PART) -> 다시 띄우면 AF_OV_CONT 로 마지막 조각 다음 프레임부터. 끝난 단계는 건너뛴다.
# 표식: FEM_<씬>_LAUNCHED, FEM_<씬>_DONE (끝), FEM_<씬>_FAIL (실패), test 는 FEM_TEST_OK / FEM_TEST_BAD
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
S=${1:?lego 또는 ficus}; G=${2:?GPU}; STAGE=${3:?test 또는 full}
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
SU=$(tr a-z A-Z <<<"$S")
U=$W/lbours/fem_$S; R=r2:storage/result/anchorflow/lbours/fem_$S; mkdir -p $U
COMMON="--no_mat --arch sgnn --gnn_layers 8 --hidden 128 --obj pts"
case $S in
  lego)  T=L000; DATA=$W/one/L000; LEN=80; GTD=$W/lbdata/test_low/L000; CFG=$W/lbdata/config.json; SHAPE=lego
         REFJ=$W/lbours/spstick/ov_L000/OVN.json; REFERR="출력만 최적화 L-BFGS 500 (spstick ov_L000)=$W/lbours/spstick/err_ov_L000.txt|Graph U-Net 풀 학습 (pool1_gunet)=$W/lbours/pool1_gunet/err_pool1_gunet_L000.txt"
         FLOOR="--bc_level node --bc_mode project --bc_softplus 0.01 --bc_stick 0.01 --bc_stick_xn"; NH=0.028284; PGF=""; TOL0=1e-2 ;;
  ficus) T=F_pg_v0; DATA=$W/lbours/ficus_v0/data/F_pg_v0; LEN=125; GTD=$W/lbours/ficus_v0/gt; CFG=$GTD/config.json; SHAPE=ficus
         REFJ=$W/lbours/ficus_v0/ov/OVN.json; REFERR="출력만 최적화 L-BFGS 500 (ficus_v0)=$W/lbours/ficus_v0/err_ov.txt"
         FLOOR="--bc_level node --bc_mode project"; NH=0.056569; PGF="--cfg_bc --pg_region_density --pg_vdamp"; TOL0=1e-3 ;;
  *) echo "[오류] 씬 $S"; exit 1 ;;
esac
REST="--inv_barrier 0 --node_h $NH --dt_cond --dt_scale --v0_traj --det_eps 0.1 --det_w 0 --det_keep_elastic --lr 3e-4 --batch 8 --det_every 200 --pool_fill $W/lbpool/fill --pool_cfg $W/lbpool/cfg --pool_combos lego_jelly --pool_drop 0.0,0.02,0.0,1.5 --pool_frames 80 --phase2 --phys_w 1.0 --save_every 100000"
OV="--out_var --ov_init predict --ov_roll 500 --ov_opt lbfgs --ov_vars out --ov_obj newton"
FEMENV="AF_FEM_TOL=${FEM_TOL:-$TOL0} AF_OV_NEWTON_STEPS=${FEM_STEPS:-100} AF_OV_NEWTON_DAMP=0 AF_OV_NEWTON_CG=${FEM_CG:-200}"
if [ $STAGE = test ]; then O=$U/test_$C; L=1; else O=$U/run; L=$LEN; fi
mkdir -p $O
up(){ rclone copy $U $R -q; }
fail(){ echo "[오류] $1 $(date)"; echo "$(date '+%m-%d %H:%M') $1" >> $U/FEM_${SU}_FAIL; up; exit 1; }
if [ -f $O/driver.hb ] && [ $(( $(date +%s) - $(cat $O/driver.hb) )) -lt 300 ]; then echo "[중단] 다른 드라이버가 돌고 있다 $(date)"; exit 0; fi
echo $$ > $O/driver.pid
( while true; do date +%s > $O/driver.hb; sleep 120; done ) & HB=$!
( while true; do sleep 300; rclone copy $U $R -q; done ) & SYNC=$!
trap 'kill $SYNC $HB 2>/dev/null; rm -f $O/driver.hb' EXIT
echo "[시작] $(date) 씬 $S 단계 $STAGE 커밋 $C 노드 $(hostname) 잡 ${PBS_JOBID:-?} GPU$G 프레임 $L"
# GPU 확인: 지정 카드에 다른 프로세스가 있으면 빌 때까지 기다린다 (다른 카드는 건드리지 않는다)
while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 500 ]; do
  echo "[대기] GPU$G 사용 중 $(date +%T)"; sleep 120; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $O/gpu_at_start.txt
B="$COMMON $FLOOR $REST $PGF"
ARGS="--infer --data $DATA --hold_traj $T --out $O/fem --tag OVN $B $OV --iters 0 --eval_t0 0 --eval_len $L --tb none"
ENVS="$FEMENV AF_FEM_LOG=$O/fem.jsonl AF_INV_LOG=$O/inv_fem.jsonl AF_ROLL_DUMP=$O/roll_fem.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 AF_ROLL_PART=$O/parts AF_ROLL_PART_EVERY=10"
# 이어 돌리기
if [ ! -f $O/roll_fem.pt ] && ls $O/parts/c*.pt > /dev/null 2>&1; then
  NS=$(ls $O/parts/c*.pt | sed 's/.*c0*\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
  ENVS="$ENVS AF_OV_CONT=$O/parts:$NS"; echo "[이어 돌리기] 조각 $NS 프레임까지 있음 -> $((NS + 1)) 부터"
fi
echo "[명령] env $ENVS CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py $ARGS" | tee -a $O/cmd.txt
if [ ! -f $O/roll_fem.pt ]; then
  ts=$(date +%m%d_%H%M); for f in v_fem.log fem.jsonl inv_fem.jsonl; do [ -f $O/$f ] && mv $O/$f $O/$f.partial_$ts; done
  echo "[실행] GPU$G $(date) 커밋 $C 단계 $STAGE" >> $U/FEM_${SU}_LAUNCHED; up
  ( # 기존 출력만 최적화 실행과 인자 대조 (OVN.json 이 생기면 바로)
    for i in $(seq 1 120); do [ -f $O/fem/OVN.json ] && break; sleep 5; done
    python - $REFJ $O/fem/OVN.json "$ARGS" <<'PY' > $O/args_vs_ref.txt 2>&1
import sys, json
a, b = (json.load(open(p))["args"] for p in sys.argv[1:3])
given = set(sys.argv[3].split())
ok = {"out", "ov_obj", "net_inertial", "eval_len"}
d = {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
# 기준 실행 뒤에 생긴 인자: 기준 json 에 키가 없고 이번 명령줄에도 안 줬으면 기본값이다 (계산 경로 그대로)
new = {k for k in d if k not in a and f"--{k}" not in given}
print(f"[인자 대조] 기준 {sys.argv[1]} vs {sys.argv[2]}")
for k, (x, y) in d.items():
    if k not in new:
        print(f"  {k}: {x!r} -> {y!r}" + ("" if k in ok else "   <-- 예상 밖"))
print(f"  (기준 실행 뒤 생긴 인자 {len(new)} 개는 명령줄에 없어 기본값: {', '.join(sorted(new))})")
bad = set(d) - ok - new
print("ARGS_OK" if not bad else f"ARGS_UNEXPECTED {sorted(bad)}")
PY
    cat $O/args_vs_ref.txt ) & AC=$!
  env $ENVS CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py $ARGS < /dev/null > $O/v_fem.log 2>&1
  wait $AC
  grep -a "롤아웃\] $T\|Traceback\|Error" $O/v_fem.log | tail -4 | cut -c1-300
  [ -f $O/roll_fem.pt ] || fail "롤아웃 실패 $(grep -a 'Traceback\|Error' $O/v_fem.log | tail -2)"
  rclone copyto $O/roll_fem.pt $R/${O#$U/}/roll_fem.pt -q
fi
grep -q ARGS_OK $O/args_vs_ref.txt || echo "[경고] 기준 인자와 예상 밖 차이 -- $O/args_vs_ref.txt"
echo "fem $(grep -a "롤아웃\] $T" $O/v_fem.log | tail -1 | grep -o '평균 [0-9.]*%')" > $O/err_fem.txt; cat $O/err_fem.txt
python - $O/fem.jsonl <<'PY' | tee $O/conv_summary.txt
import sys, json
R = [json.loads(l) for l in open(sys.argv[1])]
nc = sum(r["conv"] for r in R)
print(f"[수렴 요약] 프레임 {len(R)} 중 수렴 {nc}, 걸음 중앙 {sorted(r['iters'] for r in R)[len(R)//2] if R else '-'} 최대 {max((r['iters'] for r in R), default='-')}, "
      f"마지막 |p|∞/(h·EXT) 최대 {max((r['p_vel_last'] for r in R), default=float('nan')):.3e}, 프레임 시간 합 {sum(r['sec'] for r in R):.0f} s")
for r in R:
    if not r["conv"]:
        print(f"  미수렴 t={r['t']}: {r['why']} 걸음 {r['iters']} |p|∞/(h·EXT) {r['p_vel_last']:.3e} |g|/|g0| {r['g_rel_last']:.3e}")
PY
if [ $STAGE = test ]; then
  grep -a "\[FEM 뉴턴\]" $O/v_fem.log | cut -c1-400
  if grep -a -q "\[FEM 뉴턴\]" $O/v_fem.log && [ -f $O/roll_fem.pt ]; then touch $O/FEM_TEST_OK; echo FEM_TEST_OK; else touch $O/FEM_TEST_BAD; echo FEM_TEST_BAD; fi
  up; exit 0
fi
{ echo "[오차 비교] $S $T $LEN 프레임 (같은 지표: 롤아웃 평균 위치 오차 / EXT)"
  echo "  FEM 뉴턴 수렴 (커밋 $C): $(cat $O/err_fem.txt)"
  IFS='|' read -ra RE <<<"$REFERR"; for q in "${RE[@]}"; do echo "  ${q%%=*}: $(cat ${q#*=} 2>/dev/null)"; done; } | tee $U/err_cmp.txt
up
# 3DGS 렌더: 정답 | FEM (프레임 번호·한국어 라벨)
[ -f $O/h5_fem/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_fem.pt --traj $GTD --out $O/h5_fem > $O/h5_fem.log 2>&1
[ -f $O/fem_$S.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_fem --config $CFG --shape $SHAPE --out $O/fem_$S.mp4 \
   --frame_label 0 --caption "FEM (뉴턴 수렴)" < /dev/null > $O/r_fem.log 2>&1
[ -f $O/fem_$S.mp4 ] || fail "FEM 렌더 실패 $(tail -3 $O/r_fem.log)"
[ -f $U/gt_$S.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $GTD --config $CFG --shape $SHAPE --out $U/gt_$S.mp4 \
   --frame_label 0 --caption "정답" < /dev/null > $U/r_gt.log 2>&1
[ -f $U/gt_$S.mp4 ] || fail "정답 렌더 실패 $(tail -3 $U/r_gt.log)"
python - $U/gt_$S.mp4 $O/fem_$S.mp4 $U/cmp_$S.mp4 <<'PY' > $U/cmp.log 2>&1
import sys, numpy as np, imageio.v2 as iio
ra, rb = iio.get_reader(sys.argv[1]), iio.get_reader(sys.argv[2])
fps = ra.get_meta_data().get("fps", 30)
A, Bv = [f for f in ra], [f for f in rb]
n = max(len(A), len(Bv)); w = iio.get_writer(sys.argv[3], fps=fps, codec="libx264", quality=8, macro_block_size=1)
for i in range(n):
    a, b = A[min(i, len(A) - 1)], Bv[min(i, len(Bv) - 1)]
    hh = max(a.shape[0], b.shape[0])
    pad = lambda f: np.pad(f, ((0, hh - f.shape[0]), (0, 0), (0, 0)), constant_values=255)
    w.append_data(np.concatenate([pad(a), pad(b)], 1))
w.close(); print(f"[저장] {sys.argv[3]} 프레임 {n} (정답 {len(A)}, FEM {len(Bv)})")
PY
cat $U/cmp.log
kill $SYNC 2>/dev/null
echo "FEM_${SU}_DONE $(date) 커밋 $C $(cat $O/err_fem.txt)" > $U/FEM_${SU}_DONE; up
echo "FEM_${SU}_DONE $(date)"
