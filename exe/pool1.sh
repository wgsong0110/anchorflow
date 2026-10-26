#!/bin/bash
# lego 한 궤적 풀 학습 (사용자 10-25) -- 일반 GNN / Graph U-Net, 바닥 softplus 0.01 + 미끄러짐 방지, 관성 예측 없음, 15000 반복.
#   bash exe/pool1.sh <gnn|gunet> <GPU>     (잡 안에서만: AF_DETACH=1 bash exe/job_run.sh $W/pool1_<m>_run_boot.sh <로그>)
#   풀 학습 명령 = 예전 lego 풀 학습 spstick pool (/home/dkta/work/spstick2.sh 의 PC, 커밋 13471f4, 결과 lbours/spstick/lgvpool/pool) 이
#     실제로 실행한 인자([pool] 줄, spstick2.out) 그대로에서
#     (1) --net_inertial 하나를 뺀다 (예전 풀은 COMMON 에 --net_inertial 이 있었다)
#     (2) gunet 은 gunet5_near 와 같은 --gnn_layer unet --gunet_levels 3 --bc_feat near --bc_near_h 1.0 을 더한다
#     (3) --pool_init_traj /home/dkta/work/lbpt/test_low/L000.pt 를 더한다: 풀의 새 상태가 전부 L000 의 0 프레임(x, v, F) 하나가 된다
#         (예전 풀은 학습 궤적이 없었다 -- --pool_drop 0.0,0.02,0.0,1.5 낙하 분포에서 매번 새 시작 상태(무작위 회전·높이·속도)를
#          뽑고(--pool_start_k 0), 그 뒤는 모델 출력으로 전진. --pool_init_traj 가 있으면 scene_pool.fresh() 가 낙하 분포 대신 이 상태를 쓴다)
#   학습 루프 인자(--data lbpt/test --hold_traj 전부 --iters 15000 --save_every 500 --val_every 500 --val_n 4 --val_len 80)도 spstick2.sh train() 그대로.
#   망·바닥 인자가 sf1·다섯 궤적 실행(noinertial / gunet5_near)과 같은지 자동 대조 (차이는 풀 학습 전용 인자뿐이어야 한다).
# 순서: (0) 명령줄 대조 (1) 기본 경로 대조: 2 반복, 이 커밋 vs 기준 커밋(gnn 13471f4 / gunet cc5a559), 다르면 멈춤
#       (2) 짧은 시험 겸 이어 돌리기 시험: A = 100 반복(50 마다 저장) -> 같은 체크포인트로 200, B = 한 번에 200 -> 망·Adam·난수·풀 비교 + 롤아웃 연기 시험
#       POOL1_TEST_ONLY=1 이면 여기서 끝 (POOL1_<M>_TEST_DONE)
#       (3) 15000 반복 (500 마다 체크포인트 -> --r2 로 R2; 체크포인트가 바뀔 때마다 풀을 뺀 망만 hist/step_N.pt 로 남긴다 -- 학습 곡선용)
#       (4) L000·0004 롤아웃 80 프레임 (F float32 덤프, 뒤집힘·바닥 통과 기록) + 3DGS 영상 + psi 색(공통 고정 범위)
#       (5) 학습 곡선: hist 체크포인트마다 L000 정답 시작 80 프레임 롤아웃 위치 오차 -> hist_eval.jsonl, TB -> curves.json
# 풀 탈락 과정 기록 (사용자 10-25): 본 학습에 --pool_trace $O/pool_trace (lib/anchorflow/pool_trace.py) -- drops.jsonl(모든 탈락: 사유·나이·정체·반복),
#   500 반복마다 그 구간 탈락 중 나이 상위 2 + 무작위 1 이력, 최대 나이 갱신 때마다 그 이력 (위치·F float16 .pt). 학습 중 같은 GPU 에서
#   새 이력을 3DGS 로 렌더 (pool_trace/vid/<이름>.mp4 + .json; 변환 h5 는 노드 /tmp 에 두었다 렌더 뒤 지움 -- 원본 .pt 는 그대로).
#   (2b) 이력 시험: 120 반복(주기 20) 이력 켬, 60 -> 120 이어 돌리기 vs 한 번에, 이력 끈 같은 120 반복과 망 비교 (이력이 학습을 바꾸지 않음).
# 멈추기: touch $U/STOP_REQUEST 뒤 학습 python 만 종료 -> 드라이버가 실패 표시 없이 정리하고 끝난다 (다시 띄우면 MAIN_last.pt 에서 이어 돈다).
# 멱등: 다시 띄우면 끝난 단계는 건너뛰고 학습은 MAIN_last.pt 에서 이어 돈다. GPU 는 인자 하나만 쓴다 (다른 프로세스가 있으면 빌 때까지 기다림).
# 출력은 처음부터 R2 와 통째 동기화 (제외 없음, 5분 루프). 아무것도 지우지 않는다 (이전 시험 디렉토리는 _prev_ 로 옮긴다).
export PYTHONUTF8=1 OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
M=${1:?gnn 또는 gunet}; G=${2:?GPU}
case $M in gnn) MU=GNN; CN=p1 ;; gunet) MU=GUNET; CN=q1 ;; *) echo "[오류] 모델 $M"; exit 1 ;; esac
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
NIT=15000
NI=$W/lbours/noinertial; G5=$W/lbours/gunet5_near
WP=$W/wt/psi_d993fb5; RANGE=$W/lbours/lego_psi/range.json
L0PT=$W/lbpt/test_low/L000.pt
VAR=${POOL1_VARIANT:-}    # s32 (사용자 10-26): 무작위 폐기 끔 --pool_rdrop 0 + 풀 크기 32 --pool_size 32, 나머지 그대로 -> 출력 pool1s32_<모델>
case "$VAR" in ""|s32|s32g3|s32r|s32g3r) ;; *) echo "[오류] POOL1_VARIANT $VAR"; exit 1 ;; esac   # s32g3 (10-26): s32 + 전진 문턱 --pool_gate 1e-3; *r (10-26): --pool_init_traj 빼고 낙하 분포 시작
U=$W/lbours/pool1${VAR}_$M${POOL1_SUFFIX:-}; R=r2:storage/result/anchorflow/lbours/pool1${VAR}_$M${POOL1_SUFFIX:-};   # POOL1_SUFFIX: 시험을 본 실행과 다른 디렉토리에서 (본 실행이 돌고 있을 때)
 O=$U/pool; mkdir -p $U $O $U/hist
if [ -f $U/driver.hb ] && [ $(( $(date +%s) - $(cat $U/driver.hb) )) -lt 300 ]; then echo "[중단] 다른 드라이버가 돌고 있다 (하트비트 5 분 안) $(date)"; exit 0; fi
echo $$ > $U/driver.pid
( while true; do date +%s > $U/driver.hb; sleep 120; done ) & HB=$!
( while true; do rclone copy $U $R -q; sleep 300; done ) & SYNC=$!
trap 'kill $SYNC $HB ${SNAP:-} ${RDR:-} 2>/dev/null; rm -f $U/driver.hb' EXIT
trap 'exit 143' TERM INT     # TERM 으로도 EXIT 정리(동기화·하트비트·스냅 루프 종료)가 돌게
echo "[시작] $(date) 커밋 $C 노드 $(hostname) 잡 ${PBS_JOBID:-?} 모델 $M GPU$G 반복 $NIT"
up(){ rclone copy $U $R -q; }
bgup(){ ( rclone copyto $1 $R/${1#$U/} -q && echo "[R2 덤프 올림] ${1#$U/} $(date +%T)" ) & }
fail(){ echo "[오류] $1 $(date)"; echo "$1" > $U/POOL1_${MU}_FAIL; up; exit 1; }
rm -f $U/POOL1_${MU}_FAIL
[ -f $L0PT ] || fail "L000 궤적 없음 $L0PT"
HOLD=$(ls $W/lbpt/test | sed 's/.pt$//' | paste -sd,)
# ---- (0) 명령줄 ----
toks(){ tr ' ' '\n' <<<"$1" | grep . | LC_ALL=C sort | paste -sd' '; }
dif(){ echo "빠짐: '$(LC_ALL=C comm -23 <(tr ' ' '\n' <<<"$1" | grep . | LC_ALL=C sort) <(tr ' ' '\n' <<<"$2" | grep . | LC_ALL=C sort) | paste -sd' ')' | 더함: '$(LC_ALL=C comm -13 <(tr ' ' '\n' <<<"$1" | grep . | LC_ALL=C sort) <(tr ' ' '\n' <<<"$2" | grep . | LC_ALL=C sort) | paste -sd' ')'"; }
PC_RUN=$(grep -a -m1 '^\[pool\] ' $W/spstick2.out | sed 's/^\[pool\] //')
PC_RE=$(P=$W/lbpool bash -c "$(grep -E '^(GATE|COMMON|PCOMMON_SP|PC)=' $W/spstick2.sh); echo \"\$PC\"")
[ -n "$PC_RUN" ] && [ "$PC_RUN" = "$PC_RE" ] || fail "spstick pool 실행 인자([pool] 줄)가 spstick2.sh 원문 재구성(PC)과 다르다"
grep -q -- " --net_inertial " <<<" $PC_RUN " || fail "예전 풀 명령에 --net_inertial 이 없다 (예상과 다름)"
PC_NI="${PC_RUN/ --net_inertial / }"
GATE="--pool_gate 1e-4 --pool_gate_mode pgrad --pool_rdrop 0.01 --pool_start_k 0 --pool_stuck_max 0"
XF="--gnn_layer unet --gunet_levels 3 --bc_feat near --bc_near_h 1.0"
INIT="--pool_init_traj $L0PT"
flags_of(){ grep -a "^\[학습 명령줄\]" $1 | sed -e 's/.* --tag EV //' -e 's/ --iters 0 --eval_t0 0 --eval_len 80 --tb none$//' | sort -u; }
B_NI=$(flags_of $NI/NOINERTIAL_LAUNCHED); B_G5=$(flags_of $G5/GUNET5_LAUNCHED)
B_GNN="$PC_NI $INIT"; B_GU="$PC_NI $XF $INIT"
if [ $M = gnn ]; then B=$B_GNN; REF=$B_NI; REFN=noinertial; REFWT=$W/wt/spstick_13471f4; REFC=13471f4
else B=$B_GU; REF=$B_G5; REFN=gunet5_near; REFWT=$W/wt/gunet_cc5a559; REFC=cc5a559; fi
D1=$(dif "$PC_RUN" "$B_GNN"); D2=$(dif "$B_GNN" "$B_GU"); D3=$(dif "$REF" "$B")
[ "$D1" = "빠짐: '--net_inertial' | 더함: '--pool_init_traj $L0PT'" ] || fail "예전 풀 -> 이번 일반 GNN 차이가 예상과 다르다 ($D1)"
[ "$(LC_ALL=C comm -13 <(tr ' ' '\n' <<<"$B_GNN" | grep . | LC_ALL=C sort) <(tr ' ' '\n' <<<"$B_GU" | grep . | LC_ALL=C sort) | paste -sd' ')" = "$(toks "$XF")" ] \
  && [ -z "$(LC_ALL=C comm -23 <(tr ' ' '\n' <<<"$B_GNN" | grep . | LC_ALL=C sort) <(tr ' ' '\n' <<<"$B_GU" | grep . | LC_ALL=C sort))" ] || fail "일반 GNN -> Graph U-Net 차이가 몸통·바닥 입력(XF)만이 아니다 ($D2)"
# 망·바닥 인자 대조: 기준 다섯 궤적 실행 인자에서 '--save_every 100000'(상태 적합 전용)을 빼고, 이번 인자에서 풀 게이트·초기 상태(풀 학습 전용)를 빼면 같아야 한다
[ "$(toks "${REF/ --save_every 100000/}")" = "$(toks "$(sed -e "s# $GATE##" -e "s# $INIT##" <<<" $B")")" ] || fail "망·바닥 인자가 $REFN 과 다르다 ($D3)"
grep -q -- "--net_inertial" <<<"$B" && fail "명령줄에 --net_inertial 이 있다"
for f in "--bc_softplus 0.01" "--bc_stick 0.01" "--bc_stick_xn"; do grep -q -- " $f\( \|$\)" <<<" $B " || fail "명령줄에 '$f' 없음"; done
if [ -n "$VAR" ]; then
  BV="$(sed 's# --pool_rdrop 0.01 # --pool_rdrop 0 #' <<<"$B") --pool_size 32"
  DV=$(dif "$B" "$BV")
  [ "$DV" = "빠짐: '0.01' | 더함: '--pool_size 0 32'" ] || fail "s32 변형 차이가 '--pool_rdrop 0.01 -> 0, --pool_size 32 더함' 이 아니다 ($DV)"
  echo "[s32 변형] pool1 $M 명령 -> 무작위 폐기 끔·풀 크기 32: $DV (--pool_stuck_max 0, 만기 없음 그대로)" | tee $U/variant.txt
  if [ "$VAR" = s32g3 ] || [ "$VAR" = s32g3r ]; then
    BG="$(sed 's# --pool_gate 1e-4 # --pool_gate 1e-3 #' <<<"$BV")"; DG=$(dif "$BV" "$BG")
    [ "$DG" = "빠짐: '1e-4' | 더함: '1e-3'" ] || fail "s32g3 변형 차이가 '--pool_gate 1e-4 -> 1e-3' 이 아니다 ($DG)"
    echo "[s32g3 변형] s32 명령 -> 전진 문턱 --pool_gate 1e-4 -> 1e-3: $DG" | tee -a $U/variant.txt
    BV=$BG
  fi
  if [ "${VAR: -1}" = r ]; then
    BR="$(sed "s# --pool_init_traj $L0PT##" <<<"$BV")"; DR=$(dif "$BV" "$BR")
    [ "$DR" = "빠짐: '--pool_init_traj $L0PT' | 더함: ''" ] || fail "r 변형 차이가 '--pool_init_traj 뺌' 이 아니다 ($DR)"
    grep -q -- " --pool_drop 0.0,0.02,0.0,1.5 " <<<" $BR " && grep -q -- " --pool_start_k 0 " <<<" $BR " || fail "r 변형에 낙하 분포(--pool_drop 0.0,0.02,0.0,1.5 --pool_start_k 0)가 없다"
    echo "[r 변형] 시작 상태 L000 0 프레임 고정 -> 예전 풀과 같은 낙하 분포 (--pool_drop 0.0,0.02,0.0,1.5: 바닥 위 0~0.02, 속도 0~1.5, 무작위 회전; --pool_start_k 0 매번 새로): $DR" | tee -a $U/variant.txt
    BV=$BR
  fi
  B=$BV
fi
[ "${VAR: -1}" = r ] && INIPAT="\[풀\] 낙하 모드" || INIPAT="\[풀\] 고정 초기 상태"
{ echo "[명령줄 pool1${VAR} $M] $B"
  echo "[예전 풀 spstick pool (13471f4) 실행 인자] $PC_RUN"
  echo "[차이 예전 풀 -> pool1 gnn] $D1"
  echo "[차이 pool1 gnn -> pool1 gunet] $D2 (= 몸통·바닥 입력만)"
  echo "[차이 $REFN(다섯 궤적·sf1 과 같은 망) -> pool1 $M] $D3 (= 풀 학습 전용 인자만: 게이트·초기 상태 더함, 상태 적합 전용 --save_every 100000 빠짐)"
  echo "[풀 새 상태] 예전: --pool_drop 0.0,0.02,0.0,1.5 낙하 분포에서 매번 새로 (학습 궤적 없음, --data lbpt/test 는 전부 홀드아웃 = 검증용) -> 이번: --pool_init_traj 로 L000 0 프레임 하나"; } | tee $U/cmdline.txt
gt_of(){ [ $1 = L000 ] && echo $W/lbdata/test_low/L000 || echo $W/lbdata/test/$1; }
pt_of(){ [ $1 = L000 ] && echo $W/lbpt/test_low/L000.pt || echo $W/lbpt/test/$1.pt; }
cp -n $W/lbdata/test/0004/video.mp4 $U/gt_0004.mp4; cp -n $W/lbdata/test_low/L000/video.mp4 $U/gt_L000.mp4
ck_k(){ python -c "import torch;c=torch.load('$1',map_location='cpu',weights_only=False,mmap=True);print(c.get('step','?'))" 2>/dev/null; }
wait_gpu(){ local u; u=$(nvidia-smi -i $G --query-gpu=uuid --format=csv,noheader)
  while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u" || [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 500 ]; do
    echo "[대기] GPU$G 사용 중 $(date +%T)" >> $U/wait.log; sleep 120; done
  echo "[GPU] $G $(date '+%F %T') $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $U/gpu.log; }
tr1(){ # tr1 <출력> <반복> <저장간격> <검증간격> <로그> [--resume ..]  -- 본 실행과 같은 학습 명령 (반복·저장·검증 간격만 인자)
  local OO=$1 N=$2 E=$3 V=$4 LG=$5; shift 5
  CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py --data $W/lbpt/test --hold_traj $HOLD --out $OO --tag MAIN "$@" $B \
    --iters $N --eval_t0 0 --eval_len 1 --save_every $E --val_every $V --val_n 4 --val_len 80 > $LG 2>&1 < /dev/null; }
CMPPY=$U/cmp_ck.py; cat > $CMPPY <<'PY_'
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False); b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
d = max(float((a["net"][k].float() - b["net"][k].float()).abs().max()) for k in a["net"] if k in b["net"])
s = max(float(b["net"][k].float().abs().max()) for k in b["net"])
oa = [t for st in a["opt"]["state"].values() for t in st.values() if torch.is_tensor(t) and t.numel() > 1]
ob = [t for st in b["opt"]["state"].values() for t in st.values() if torch.is_tensor(t) and t.numel() > 1]
o = max(float((x - y).abs().max()) for x, y in zip(oa, ob)) if oa and len(oa) == len(ob) else float("nan")
pa, pb = (a.get("pool") or {}).get("items") or [], (b.get("pool") or {}).get("items") or []
na, nb = sum(q is not None for q in pa), sum(q is not None for q in pb)
px = float("nan")
if na and na == nb:
    px = max(float((qa["x"].float() - qb["x"].float()).abs().max()) for qa, qb in zip(pa, pb) if qa is not None and qb is not None)
fz = sum("fuse" in k for k in a["net"])
print(f"반복 {a['step']}/{b['step']}  입력폭 {a['n_feat']}/{b['n_feat']}  키 같음 {sorted(a['net']) == sorted(b['net'])}  스킵 결합 키 {fz}  "
      f"망 파라미터 최대차 {d:.3e} (최대 크기 {s:.3e})  Adam 상태 최대차 {o:.3e}  생성기 같음 {bool((a['rng_gen'] == b['rng_gen']).all())}  "
      f"풀 상태 {na}/{nb} 위치 최대차 {px:.3e}")
PY_
maxd(){ python -c "import re,sys;m=re.search(r'망 파라미터 최대차 ([0-9.e+-]+)', sys.argv[1]);sys.exit(0 if m and float(m.group(1))<$2 else 1)" "$1"; }
NFW=33; [ $M = gunet ] && NFW=31
render_trace(){ # render_trace <이력 .pt> <vid 디렉토리> -- 3DGS 렌더 (프레임 0 = 풀 시작 = L000 0 프레임; 이력 시작이 0 이 아니면 덤프 첫 프레임부터)
  local f=$1 V=$2 n; n=$(basename $1 .pt); mkdir -p $V
  [ -f $V/$n.mp4 ] && return 0
  local H=/tmp/pool1_h5_${M}${VAR}_$n;   # 변형 이름까지 (같은 노드에서 변형들이 같은 이름 이력을 동시에 렌더해 서로 지웠다)
  [ -e $H ] && mv $H ${H}_prev_$(date +%s)
  local FF; FF=$(python -c "import torch,json;m=torch.load('$f',map_location='cpu',weights_only=False);d=dict(m['meta'],frames=m['frames']);json.dump(d,open('$V/$n.json','w'),ensure_ascii=False);print('--first_from_dump' if d['trace_from']>0 or d.get('has_start') else '')") || return 1
  local TJ=$W/lbdata/test_low/L000
  if python -c "import json,sys;sys.exit(0 if 'R' in json.load(open('$V/$n.json')) else 1)"; then
    # 낙하 분포 시작(r 변형): 렌더용 회전 R 은 그 상태의 시작 자세 것 (meta.json 만 바꾼 임시 궤적 디렉토리, h5 와 함께 지움)
    TJ=$H.traj; mkdir -p $TJ; python -c "import json;m=json.load(open('$W/lbdata/test_low/L000/meta.json'));m['R']=json.load(open('$V/$n.json'))['R'];json.dump(m,open('$TJ/meta.json','w'))" || return 1
  fi
  python -u exe/lb_roll_to_h5.py --dump $f --traj $TJ --out $H $FF > $V/$n.h5.log 2>&1 \
    && CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $H --config $W/lbdata/config.json --shape lego --out $V/$n.mp4 > $V/$n.r.log 2>&1 < /dev/null
  local rc=$?; rm -rf $H $H.traj     # 변환 h5 (float32 다시 쓴 것) 만 지운다 -- 원본 이력 .pt 는 그대로
  [ $rc = 0 ] && [ -f $V/$n.mp4 ] || { touch $V/$n.fail; return 1; }; }
render_loop(){ while true; do for f in $(ls $O/pool_trace/*.pt 2>/dev/null); do n=$(basename $f .pt); [ -f $O/pool_trace/vid/$n.mp4 ] || [ -f $O/pool_trace/vid/$n.fail ] || { render_trace $f $O/pool_trace/vid && echo "[이력 렌더] $n $(date +%T)"; }; done; sleep 120; done; }
# ---- (1) 기본 경로 대조: 같은 명령 2 반복 -- 이 커밋 vs 기준 커밋 ----
if [ ! -f $U/base_equiv_$C.ok ]; then
  wait_gpu; echo "[기본 경로 대조] GPU$G $(date +%T)"
  BE=$U/base_equiv_$C; [ -e $BE ] && mv $BE ${BE}_prev_$(date +%H%M%S); mkdir -p $BE
  [ -d $REFWT ] || fail "기준 워크트리 없음 $REFWT"
  tr1 $BE/new 2 1 0 $BE/new.log
  ( cd $REFWT && tr1 $BE/old 2 1 0 $BE/old.log )
  TB=$(cat $BE/new.log $BE/old.log | grep -a -c Traceback)
  CMPB=$(python $CMPPY $BE/new/MAIN_last.pt $BE/old/MAIN_last.pt 2>&1 | tail -1)
  INI=$(grep -a -m1 "$INIPAT" $BE/new.log | cut -c1-200)
  echo "[기본 경로 대조 $M] 이 커밋 $C vs 기준 커밋 $REFC (2 반복): $CMPB | Traceback $TB | $INI" | tee $U/base_equiv.txt
  [ "$TB" = 0 ] && grep -q "입력폭 $NFW/$NFW  키 같음 True" <<<"$CMPB" && [ -n "$INI" ] || fail "기본 경로 대조 실패 (Traceback·입력폭·키·고정 초기 상태)"
  maxd "$CMPB" 1e-4 || fail "기본 경로 망 파라미터 차이 1e-4 이상"
  touch $U/base_equiv_$C.ok; up
fi
# ---- (2) 짧은 시험 겸 이어 돌리기 시험: A = 100 반복(50 마다 저장, 100 마다 검증) -> 같은 체크포인트로 200, B = 한 번에 200 ----
if [ ! -f $U/resume_$C.ok ]; then
  wait_gpu; echo "[짧은 시험·이어 돌리기 시험] GPU$G $(date +%T)"
  RT=$U/resume_$C/test; RB=$U/resume_$C/ref; [ -e $U/resume_$C ] && mv $U/resume_$C $U/resume_${C}_prev_$(date +%H%M%S); mkdir -p $RT $RB
  t0=$(date +%s); tr1 $RT 100 50 100 $RT/r100.log; t1=$(date +%s); tr1 $RT 200 50 100 $RT/r200.log --resume $RT/MAIN_last.pt; t2=$(date +%s)
  tr1 $RB 200 50 100 $RB/r200.log; t3=$(date +%s)
  K=$(ck_k $RT/MAIN_last.pt); RS=$(grep -a "\[재개\]" $RT/r200.log | cut -c1-140 | tr '\n' ' ')
  TB=$(cat $RT/r100.log $RT/r200.log $RB/r200.log | grep -a -c Traceback)
  CMP=$(python $CMPPY $RT/MAIN_last.pt $RB/MAIN_last.pt 2>&1 | tail -1)
  INI=$(grep -a -m1 "$INIPAT" $RB/r200.log | cut -c1-200)
  VL=$(grep -a "\[검증 [0-9]*\]" $RB/r200.log | sed 's/ *(best.*//; s/비 [0-9.]* //' | cut -c1-80 | tr '\n' ' ')
  { echo "[이어 돌리기 $M] 200 실행: $RS | 마지막 반복 $K | Traceback $TB | 이어 돈 것 vs 한 번에: $CMP"
    echo "[시간] 100 반복 $((t1-t0)) s, 이어 100 반복 $((t2-t1)) s, 한 번에 200 반복 $((t3-t2)) s (검증 포함) | $INI"
    echo "[짧은 시험 200 반복 검증 목적함수] $VL"; } | tee $U/resume_test.txt
  [ "$K" = 200 ] && grep -q "step 100" <<<"$RS" && grep -q "풀 [0-9]* 개 상태 이어받음" <<<"$RS" && [ "$TB" = 0 ] && grep -q "입력폭 $NFW/$NFW " <<<"$CMP" && [ -n "$INI" ] || fail "이어 돌리기 확인 실패 -- 학습 안 함"
  [ $M = gunet ] && grep -q "스킵 결합 키 0 " <<<"$CMP" && fail "Graph U-Net 스킵 결합 키 없음"
  if ! maxd "$CMP" 1e-3; then
    # 전진 판정(pgrad <= 문턱)이 문턱 근처에 몰리면 GPU 비결정성(1e-6 수준)만으로 판정이 갈려 풀 궤적이 갈라진다 (s32g3 문턱 1e-3 실측).
    # 이어 돌리기 탓인지 가리려고 같은 200 반복을 한 번 더 한 번에 돌려 비교: 한 번에 두 번 돈 차이가 같은 크기면 이어 돌리기 문제가 아니다.
    RB2=$U/resume_$C/ref2; mkdir -p $RB2; tr1 $RB2 200 50 100 $RB2/r200.log
    CMP2=$(python $CMPPY $RB/MAIN_last.pt $RB2/MAIN_last.pt 2>&1 | tail -1)
    echo "[비결정성 확인] 한 번에 200 두 번 비교: $CMP2" | tee -a $U/resume_test.txt
    python - "$CMP" "$CMP2" <<'PY_' || fail "이어 돈 것과 한 번에 돈 것의 망 차이가 1e-3 이상이고, 한 번에 두 번 돈 차이보다 10 배 넘게 크다"
import re, sys
d1 = float(re.search(r"망 파라미터 최대차 ([0-9.e+-]+)", sys.argv[1]).group(1))
d2 = float(re.search(r"망 파라미터 최대차 ([0-9.e+-]+)", sys.argv[2]).group(1))
print(f"[비결정성 확인] 이어 돈 것 vs 한 번에 {d1:.3e}, 한 번에 두 번 {d2:.3e} -> {'같은 크기 (판정 경계 비결정성, 이어 돌리기 문제 아님)' if d1 <= 10 * d2 else '이어 돌리기 쪽이 훨씬 큼'}")
sys.exit(0 if d1 <= 10 * d2 else 1)
PY_
  fi
  for T in L000 0004; do
    env AF_INV_LOG=$RT/inv_$T.jsonl AF_TUNNEL_LOG=$RT/tun_$T.jsonl AF_ROLL_DUMP=$RT/roll_$T.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$G \
      python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $RT/roll_$T --tag EV --resume $RT/MAIN_last.pt $B --iters 0 --eval_t0 0 --eval_len 3 --tb none > $RT/roll_$T.log 2>&1 < /dev/null
    RL=$(grep -a "롤아웃\] $T" $RT/roll_$T.log | tail -1 | sed 's/ (정지.*//' | cut -c1-160); echo "[롤아웃 연기 시험 $T] $RL | Traceback $(grep -a -c Traceback $RT/roll_$T.log) | 덤프 $(ls -la $RT/roll_$T.pt 2>/dev/null | awk '{print $5}') B | 뒤집힘 기록 $(wc -l < $RT/inv_$T.jsonl 2>/dev/null) 줄" | tee -a $U/resume_test.txt
    [ -n "$RL" ] && [ "$(grep -a -c Traceback $RT/roll_$T.log)" = 0 ] && [ -f $RT/roll_$T.pt ] || fail "롤아웃 연기 시험 실패 ($T) -- 학습 안 함"
  done
  touch $U/resume_$C.ok; up
fi
# ---- (2b) 풀 이력 시험 ----
if [ ! -f $U/trace_$C.ok ]; then
  wait_gpu; echo "[이력 시험] GPU$G $(date +%T)"
  TT=$U/trace_$C; [ -e $TT ] && mv $TT ${TT}_prev_$(date +%H%M%S); mkdir -p $TT/a $TT/b $TT/c
  TRF="--pool_trace_period 20"
  tr1 $TT/a 60 30 0 $TT/a60.log --pool_trace $TT/a/pool_trace $TRF; tr1 $TT/a 120 30 0 $TT/a120.log --resume $TT/a/MAIN_last.pt --pool_trace $TT/a/pool_trace $TRF
  tr1 $TT/b 120 30 0 $TT/b120.log --pool_trace $TT/b/pool_trace $TRF
  tr1 $TT/c 120 30 0 $TT/c120.log
  TB=$(cat $TT/*.log | grep -a -c Traceback)
  CAB=$(python $CMPPY $TT/a/MAIN_last.pt $TT/b/MAIN_last.pt 2>&1 | tail -1); CBC=$(python $CMPPY $TT/b/MAIN_last.pt $TT/c/MAIN_last.pt 2>&1 | tail -1)
  ND=$(cat $TT/b/pool_trace/drops.jsonl 2>/dev/null | wc -l); NS=$(cat $TT/b/pool_trace/saved.jsonl 2>/dev/null | wc -l)
  RS=$(grep -a "\[풀 이력\]" $TT/a120.log | head -1 | cut -c1-160)
  SZ=$(du -sh $TT/b/pool_trace | cut -f1)
  { echo "[이력 시험 $M] Traceback $TB | 탈락 기록 $ND 줄 (사유 $(python -c "import json,collections;print(dict(collections.Counter(json.loads(l)['reason'] for l in open('$TT/b/pool_trace/drops.jsonl'))))" 2>/dev/null)) | 저장 이력 $NS 개, 디렉토리 $SZ"
    echo "[이력 시험 저장] $(python -c "import json;[print(r['file'],r['n_trace'],'프레임',round(r['bytes']/1e6,1),'MB') for r in map(json.loads,open('$TT/b/pool_trace/saved.jsonl'))]" 2>/dev/null | tr '\n' ';')"
    echo "[이력 켬 60 -> 120 이어 vs 한 번에] $CAB"
    echo "[이력 켬 vs 끔 (120 반복)] $CBC"
    echo "[이어 돌릴 때 이력] $RS | 이어 돈 쪽 탈락 기록 $(cat $TT/a/pool_trace/drops.jsonl 2>/dev/null | wc -l) 줄"; } | tee $U/trace_test.txt
  NP=$(grep -a -c "\[풀 이력\] 반복" $TT/b120.log)
  echo "[이력 시험 주기 저장] 주기 경계 저장 줄 $NP 개 (120/20 = 6 이어야): $(grep -a "\[풀 이력\] 반복" $TT/b120.log | tail -2 | tr '\n' ' ')" | tee -a $U/trace_test.txt
  [ "$TB" = 0 ] && [ "$ND" -gt 0 ] && [ "$NS" -gt 0 ] && [ -n "$RS" ] && [ "$NP" = 6 ] || fail "이력 시험 실패 (Traceback·탈락 기록·저장 이력·이어받기·주기 저장)"
  # 문턱: 같은 설정 두 번 돌린 차이(GPU 원자 연산 비결정성, 120 반복에 1e-6~2e-5)보다 크게 -- b085da6 gnn 실측: 이력 켬 두 실행 차 1.67e-05, 켬/끔 차 1.75e-05
  if ! maxd "$CBC" 1e-3 || ! maxd "$CAB" 1e-3; then
    # 전진 판정 경계 비결정성 (이어 돌리기 시험과 같은 사정, s32g3 문턱 1e-3 실측): 이력 켜고 한 번에 120 을 한 번 더 돌려 그 차이와 비교
    tr1 $TT/d 120 30 0 $TT/d120.log --pool_trace $TT/d/pool_trace $TRF
    CBD=$(python $CMPPY $TT/b/MAIN_last.pt $TT/d/MAIN_last.pt 2>&1 | tail -1)
    echo "[비결정성 확인] 이력 켬 한 번에 120 두 번 비교: $CBD" | tee -a $U/trace_test.txt
    python - "$CAB" "$CBC" "$CBD" <<'PY_' | tee -a $U/trace_test.txt; [ "${PIPESTATUS[0]}" = 0 ] || fail "이력 켬/끔 또는 이어 돌리기 차이가 1e-3 이상이고 한 번에 두 번 돈 차이보다 10 배 넘게 크다"
import re, sys
f = lambda t: float(re.search(r"망 파라미터 최대차 ([0-9.e+-]+)", t).group(1))
ab, bc, bd = f(sys.argv[1]), f(sys.argv[2]), f(sys.argv[3])
ok = ab <= max(1e-3, 10 * bd) and bc <= max(1e-3, 10 * bd)
print(f"[비결정성 확인] 이어 돈 것 vs 한 번에 {ab:.3e}, 이력 켬 vs 끔 {bc:.3e}, 같은 설정 한 번에 두 번 {bd:.3e} -> "
      f"{'같은 크기 (판정 경계 비결정성)' if ok else '비결정성보다 10 배 넘게 큼'}")
sys.exit(0 if ok else 1)
PY_
  fi
  # 저장 이력 하나를 실제로 3DGS 로 렌더해 본다
  f=$(ls $TT/b/pool_trace/*.pt | head -1); render_trace $f $TT/b/pool_trace/vid && ls -la $TT/b/pool_trace/vid | tee -a $U/trace_test.txt || fail "이력 렌더 시험 실패"
  touch $U/trace_$C.ok; up
fi
if [ -n "${POOL1_TEST_ONLY:-}" ]; then touch $U/POOL1_${MU}_TEST_DONE; up; echo "POOL1_${MU}_TEST_DONE"; exit 0; fi
# ---- (3) 학습 15000 반복 ----
snap(){ # MAIN_last.pt 가 바뀔 때마다 풀·크리틱을 뺀 망 체크포인트를 hist/step_N.pt 로 (학습 곡선용; 원본은 그대로)
  while true; do
    if [ -f $O/MAIN_last.pt ] && [ $O/MAIN_last.pt -nt $U/hist/.last ]; then
      s0=$(stat -c %s $O/MAIN_last.pt); sleep 30
      if [ "$(stat -c %s $O/MAIN_last.pt)" = "$s0" ]; then
        touch -r $O/MAIN_last.pt $U/hist/.cand
        python - $O/MAIN_last.pt $U/hist <<'PY' && mv $U/hist/.cand $U/hist/.last
import os, sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
p = os.path.join(sys.argv[2], f"step_{int(c['step']):06d}.pt")
if not os.path.exists(p):
    c = {k: v for k, v in c.items() if k not in ("pool", "critic", "opt_c")}
    torch.save(c, p + ".tmp"); os.replace(p + ".tmp", p); print("[망 기록]", p, flush=True)
PY
      fi
    fi
    sleep 60; done; }
K=$(ck_k $O/MAIN_last.pt)
if [ "$K" != $NIT ] || ! grep -a -q DEFORM_OK $U/pool.log 2>/dev/null; then
  wait_gpu
  snap >> $U/snap.log 2>&1 & SNAP=$!
  render_loop >> $U/render.log 2>&1 & RDR=$!
  RES=''; [ -f $O/MAIN_last.pt ] && RES="--resume $O/MAIN_last.pt"
  CMD="python -u exe/train_deform.py --data $W/lbpt/test --hold_traj $HOLD --out $O --tag MAIN $RES $B --iters $NIT --eval_t0 0 --eval_len 80 --save_every 500 --val_every 500 --val_n 4 --val_len 80 --tb $U/tb --r2 $R/pool --pool_trace $O/pool_trace"
  echo "[학습 시작] pool1 $M: 커밋 $C, GPU$G, 이어 돌기 ${K:-없음}, $(date '+%F %T')" | tee -a $U/POOL1_LAUNCHED
  echo "[학습 명령줄] CUDA_VISIBLE_DEVICES=$G $CMD" | tee -a $U/POOL1_LAUNCHED; up
  CUDA_VISIBLE_DEVICES=$G $CMD >> $U/pool.log 2>&1 < /dev/null
  if [ -f $U/STOP_REQUEST ]; then mv $U/STOP_REQUEST $U/STOP_REQUEST.done_$(date +%m%d_%H%M%S); echo "[멈춤 요청] 학습 중단 -- 다시 띄우면 MAIN_last.pt 에서 이어 돈다 $(date)"; kill $SNAP $RDR 2>/dev/null; up; exit 0; fi
  K=$(ck_k $O/MAIN_last.pt); echo "[학습 끝] 반복 $K $(date +%T)"; sleep 120; kill $SNAP $RDR 2>/dev/null; up
  [ "$K" = $NIT ] && grep -a -q DEFORM_OK $U/pool.log || fail "$NIT 못 미침 (반복 $K) -- 롤아웃 생략"
fi
for f in $(ls $O/pool_trace/*.pt 2>/dev/null); do render_trace $f $O/pool_trace/vid > /dev/null 2>&1; done; up
# ---- (4) L000·0004 롤아웃 + 영상 + psi (sf1 과 같은 명령·도구) ----
roll(){ T=$1; N=pool1_${M}_$T
  if [ ! -f $U/roll_$N.pt ] || ! grep -a -q "롤아웃\] $T" $U/v_$N.log 2>/dev/null; then
    for f in $U/roll_$N.pt $U/inv_$N.jsonl $U/tun_$N.jsonl $U/v_$N.log; do [ -e $f ] && mv $f $f.prev_$(date +%H%M%S); done
    echo "[롤아웃 명령] AF_INV_LOG=$U/inv_$N.jsonl AF_TUNNEL_LOG=$U/tun_$N.jsonl AF_ROLL_DUMP=$U/roll_$N.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $U/v_$N --tag EV --resume $O/MAIN_last.pt $B --iters 0 --eval_t0 0 --eval_len 80 --tb none" > $U/cmd_roll_$T.txt
    env AF_INV_LOG=$U/inv_$N.jsonl AF_TUNNEL_LOG=$U/tun_$N.jsonl AF_ROLL_DUMP=$U/roll_$N.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$G \
      python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $U/v_$N --tag EV --resume $O/MAIN_last.pt $B --iters 0 --eval_t0 0 --eval_len 80 --tb none > $U/v_$N.log 2>&1 < /dev/null
    bgup $U/roll_$N.pt; fi
  grep -a "롤아웃\] $T\|Traceback\|Error" $U/v_$N.log | tail -2 | sed 's/ (정지.*//' | cut -c1-200
  [ -f $U/roll_$N.pt ] || { echo "[오류] $T 롤아웃 덤프 없음"; return 1; }
  echo "$N $(grep -a '롤아웃\] ' $U/v_$N.log | tail -1 | grep -o '평균 [0-9.]*%')" > $U/err_$N.txt; up
  [ -f $U/h5_$N/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $U/roll_$N.pt --traj $(gt_of $T) --out $U/h5_$N > $U/h5_$N.log 2>&1
  [ -f $U/$N.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $U/h5_$N --config $W/lbdata/config.json --shape lego --out $U/$N.mp4 > $U/r_$N.log 2>&1 < /dev/null
  [ -f $U/$N.mp4 ] || { echo "[오류] $T 렌더"; return 1; }
  up
  D=$U/psi/$T; mkdir -p $D
  ( cd $WP && { [ -f $D/col_$CN.npy ] || CUDA_VISIBLE_DEVICES=$G python -u exe/psi_color_lb.py --gt_dir $(gt_of $T) --pt $(pt_of $T) --dump $CN=$U/roll_$N.pt --out $D \
        --eps 1e-3 --q 1 99 --cmap turbo --range $RANGE --stats_name defstats_$T.txt --stats_frames 1 40 80 > $D/psi_color.log 2>&1 < /dev/null; }
    grep -q PSI_COLOR_DONE $D/psi_color.log || { tail -5 $D/psi_color.log; exit 1; }
    [ -f $D/h5_$CN/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $U/roll_$N.pt --traj $(gt_of $T) --out $D/h5_$CN > $D/h5_$CN.log 2>&1
    [ -f $D/psi_${CN}_lego.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $D/h5_$CN --config $W/lbdata/config.json --shape lego \
       --colors $D/col_$CN.npy --out $D/psi_${CN}_lego.mp4 > $D/r_$CN.log 2>&1 < /dev/null
    [ -f $D/psi_${CN}_lego.mp4 ] ) || { echo "[오류] $T psi"; return 1; }
  grep -a "\[색 범위\]" $D/psi_color.log; cat $D/defstats_$T.txt
  cp $NI/psi/$T/psi_gt_lego.mp4 $D/psi_gt_lego.mp4; cp $NI/psi/$T/range.json $D/range_noinertial.json
  touch $U/.done_roll_$T; }
for T in L000 0004; do
  [ -f $U/.done_roll_$T ] || { wait_gpu; roll $T > $U/task_roll_$T.log 2>&1; cat $U/task_roll_$T.log | cut -c1-200; }
  [ -f $U/.done_roll_$T ] || fail "롤아웃·영상·psi 단계 실패 ($T)"
done
cat $U/err_pool1_${M}_L000.txt $U/err_pool1_${M}_0004.txt > $U/errors.txt
# ---- (5) 학습 곡선: hist 망마다 L000 정답 시작 80 프레임 롤아웃 위치 오차 ----
wait_gpu
for f in $(ls $U/hist/step_*.pt 2>/dev/null); do s=$(basename $f .pt | sed 's/step_0*//')
  grep -q "\"step\": $s," $U/hist_eval.jsonl 2>/dev/null && continue
  AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py --infer --data $W/one/L000 --hold_traj L000 --out $U/hist_ev/s$s --tag EV --resume $f $B \
    --iters 0 --eval_t0 0 --eval_len 80 --tb none > $U/hist_ev_$s.log 2>&1 < /dev/null
  e=$(grep -a "롤아웃\] L000" $U/hist_ev_$s.log | tail -1 | grep -o '평균 [0-9.]*%' | grep -o '[0-9.]*')
  [ -n "$e" ] && echo "{\"step\": $s, \"err_L000\": $e}" >> $U/hist_eval.jsonl && echo "[곡선] 반복 $s L000 $e%"
done
python - $U/tb $U/hist_eval.jsonl $U/curves.json $M <<'PY'
import glob, json, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
out = {}
for d in sorted(glob.glob(sys.argv[1] + "/*")) or [sys.argv[1]]:
    ea = EventAccumulator(d, size_guidance={"scalars": 0}); ea.Reload()
    for k in ea.Tags()["scalars"]:
        if k.split("/")[0] in ("학습", "검증", "풀", "게이트", "속도"):
            out.setdefault(k, []).extend((e.step, e.value) for e in ea.Scalars(k))
H = sorted((json.loads(l) for l in open(sys.argv[2]) if l.strip()), key=lambda r: r["step"])
out["L000롤아웃_위치오차%"] = [(r["step"], r["err_L000"]) for r in H]
json.dump({"pool1_" + sys.argv[4]: out}, open(sys.argv[3], "w"), ensure_ascii=False)
print("곡선", {k: len(v) for k, v in out.items() if k.startswith("학습/") or "롤아웃" in k})
PY
up
kill $SYNC 2>/dev/null
rclone copy $U $R -q && echo "[R2 업로드 완료] $(date)" && touch $U/ALL_DONE $U/POOL1_${MU}_CLUSTER_DONE && rclone copy $U $R -q
rclone check $U $R --one-way --size-only -q && echo "[R2 대조] 전부 올라감"
echo POOL1_${MU}_CLUSTER_DONE
