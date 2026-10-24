#!/bin/bash
# lego L000 한 궤적 상태 적합 (사용자 10-24) -- 바닥 softplus 0.01 + 미끄러짐 방지(--bc_stick 0.01 --bc_stick_xn), 관성 예측 없음.
#   bash exe/sf1.sh <gnn|gunet> <GPU>      (잡 안에서만: AF_DETACH=1 bash exe/job_run.sh $W/sf1_<m>_boot.sh <로그>)
#   망·바닥·손실 인자 = 다섯 궤적 두 실행이 **실제로 실행한 학습 명령줄**에서 그대로 꺼낸다 (스크립트 원문 재구성과도 대조):
#     gnn   : lbours/noinertial  (exe/noinertial.sh, 5c2e3a2)       일반 GNN 메시지 패싱 8 층 128, 바닥 입력 = 모든 노드 거리+법선
#     gunet : lbours/gunet5_near (GUNET5_TAG=near exe/gunet5.sh, cc5a559) + --gnn_layer unet --gunet_levels 3 --bc_feat near --bc_near_h 1.0
#   데이터만 L000 한 궤적: 예전 한 궤적 학습(spfloor.sh 의 sf)과 같은 방식 -- AF_STATE_FIT=<L000 상태 덤프>:15000:8, --data one/L000 --hold_traj L000.
#     상태 덤프 = 다섯 궤적 두 실행이 쓴 L000 덤프 그대로 (lbours/spstick/roll_ov_L000.pt). 반복 15000, 배치 8, 손실 증분 포텐셜(ip), 500 마다 체크포인트·검증.
# 순서: (0) 명령줄 대조 (1) 기본 경로 대조: 1 반복, 이 커밋 vs 기준 실행 커밋 (다르면 멈춤) + 같은 1 반복을 다궤적 경로(L000 하나)로 (참고)
#       (2) 짧은 시험 겸 이어 돌리기 시험: A = 200 반복(100 마다 저장) 뒤 같은 체크포인트로 300 까지, B = 한 번에 300 -> 망·Adam·난수 비교, 롤아웃 연기 시험 L000·0004
#       SF1_TEST_ONLY=1 이면 여기서 끝 (SF1_<M>_TEST_DONE)
#       (3) 15000 반복 (500 마다 체크포인트 -> R2 바로) (4) L000·0004 롤아웃 80 프레임 (F float32 덤프, 뒤집힘·바닥 통과 기록) + 3DGS 영상 + psi 색(공통 고정 범위)
#       (5) 학습 곡선 json (이번 + 다섯 궤적 두 실행)
# 멱등: 다시 띄우면 끝난 단계는 건너뛰고 학습은 체크포인트에서 이어 돈다. GPU 는 인자 하나만 쓴다 (다른 프로세스가 있으면 빌 때까지 기다림).
# 출력은 처음부터 R2 와 통째 동기화 (제외 없음, 5분 루프). 아무것도 지우지 않는다 (이전 시험 디렉토리는 _prev_ 로 옮긴다).
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
M=${1:?gnn 또는 gunet}; G=${2:?GPU}
case $M in gnn) MU=GNN; CN=g1 ;; gunet) MU=GUNET; CN=u1 ;; *) echo "[오류] 모델 $M"; exit 1 ;; esac
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
NIT=15000
NI=$W/lbours/noinertial; G5=$W/lbours/gunet5_near; S=$W/lbours/spstick
WP=$W/wt/psi_d993fb5; RANGE=$W/lbours/lego_psi/range.json
DUMP=$S/roll_ov_L000.pt
U=$W/lbours/sf1_$M; R=r2:storage/result/anchorflow/lbours/sf1_$M; mkdir -p $U
if [ -f $U/driver.hb ] && [ $(( $(date +%s) - $(cat $U/driver.hb) )) -lt 300 ]; then echo "[중단] 다른 드라이버가 돌고 있다 (하트비트 5 분 안) $(date)"; exit 0; fi
echo $$ > $U/driver.pid
( while true; do date +%s > $U/driver.hb; sleep 120; done ) & HB=$!
( while true; do rclone copy $U $R -q; sleep 300; done ) & SYNC=$!
trap 'kill $SYNC $HB 2>/dev/null' EXIT
echo "[시작] $(date) 커밋 $C 노드 $(hostname) 잡 ${PBS_JOBID:-?} 모델 $M GPU$G 반복 $NIT"
up(){ rclone copy $U $R -q; }
bgup(){ ( rclone copyto $1 $R/${1#$U/} -q && echo "[R2 덤프 올림] ${1#$U/} $(date +%T)" ) & }
fail(){ echo "[오류] $1 $(date)"; echo "$1" > $U/SF1_${MU}_FAIL; up; exit 1; }
rm -f $U/SF1_${MU}_FAIL
[ -f $DUMP ] || fail "L000 상태 덤프 없음 $DUMP"
# ---- (0) 명령줄: 기준 실행이 실제로 실행한 학습 명령줄에서 '--tag EV ' 와 ' --iters 0' 사이 (망·바닥·손실 인자) ----
flags_of(){ grep -a "^\[학습 명령줄\]" $1 | sed -e 's/.* --tag EV //' -e 's/ --iters 0 --eval_t0 0 --eval_len 80 --tb none$//' | sort -u; }
XF="--gnn_layer unet --gunet_levels 3 --bc_feat near --bc_near_h 1.0"
B_NI=$(flags_of $NI/NOINERTIAL_LAUNCHED); B_G5=$(flags_of $G5/GUNET5_LAUNCHED)
[ "$(wc -l <<<"$B_NI")" = 1 ] && [ "$(wc -l <<<"$B_G5")" = 1 ] && [ -n "$B_NI" ] || fail "기준 학습 명령줄이 하나가 아니다"
B_RE=$(P=$W/lbpool bash -c "$(grep -E '^(COMMON|REST_OLD|FLOOR_SP|REST|BASE_SP|BASE_NOXN|BASE|BASE_NI)=' $W/wt/noinertial_5c2e3a2/exe/noinertial.sh); echo \"\$BASE_NI\"")
[ "$B_RE" = "$B_NI" ] || fail "noinertial 실행 명령줄이 noinertial.sh 원문 재구성(BASE_NI)과 다르다"
[ "$B_G5" = "$B_NI $XF" ] || fail "gunet5_near 실행 명령줄이 noinertial + XF 가 아니다"
if [ $M = gnn ]; then B=$B_NI; REFWT=$W/wt/noinertial_5c2e3a2; REFC=5c2e3a2; else B=$B_G5; REFWT=$W/wt/gunet_cc5a559; REFC=cc5a559; fi
for f in "--bc_softplus 0.01" "--bc_stick 0.01" "--bc_stick_xn" "--bc_mode project" "--bc_level node"; do
  grep -q -- " $f\( \|$\)" <<<" $B " || fail "명령줄에 '$f' 없음"; done
grep -q -- "--net_inertial" <<<"$B" && fail "명령줄에 --net_inertial 이 있다"
{ echo "[명령줄 $M] $B"
  echo "[대조] = 기준 실행 $( [ $M = gnn ] && echo noinertial || echo gunet5_near ) 의 실행 학습 명령줄 인자 그대로 (noinertial.sh 원문 재구성과도 같음); --bc_softplus 0.01 --bc_stick 0.01 --bc_stick_xn 있음, --net_inertial 없음"
  echo "[바뀐 것] AF_STATE_FIT_MULTI=<다섯 궤적>:15000:8 -> AF_STATE_FIT=$DUMP:15000:8 | --data $W/multi5 -> $W/one/L000 | --hold_traj L000,0000,0001,0002,0003 -> L000 | 출력 lbours/sf1_$M"; } | tee $U/cmdline.txt
gt_of(){ [ $1 = L000 ] && echo $W/lbdata/test_low/L000 || echo $W/lbdata/test/$1; }
pt_of(){ [ $1 = L000 ] && echo $W/lbpt/test_low/L000.pt || echo $W/lbpt/test/$1.pt; }
cp -n $W/lbdata/test/0004/video.mp4 $U/gt_0004.mp4; cp -n $W/lbdata/test_low/L000/video.mp4 $U/gt_L000.mp4
ck_k(){ python -c "import torch;c=torch.load('$1',map_location='cpu',weights_only=False);print(c.get('k',c.get('it',c.get('step','?'))))" 2>/dev/null; }
wait_gpu(){ # 인자 카드에 다른 프로세스가 없고 메모리 500 MiB 미만일 때까지
  local u; u=$(nvidia-smi -i $G --query-gpu=uuid --format=csv,noheader)
  while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u" || [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 500 ]; do
    echo "[대기] GPU$G 사용 중 $(date +%T)" >> $U/wait.log; sleep 120; done
  echo "[GPU] $G $(date '+%F %T') $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $U/gpu.log; }
fit(){ # fit <출력디렉토리> <반복> <저장간격> <로그> <검증 프레임>  -- 본 실행과 같은 명령 (반복·저장간격·검증 길이만 인자)
  local O=$1 N=$2 E=$3 LG=$4 EL=$5
  env AF_STATE_FIT=$DUMP:$N:8 AF_STATE_FIT_LOSS=ip AF_STATE_FIT_EVERY=$E AF_STATE_FIT_CK=$O/ck.pt AF_STATE_FIT_TB=$O/tb \
    CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py --infer --data $W/one/L000 --hold_traj L000 --out $O/run --tag EV $B --iters 0 --eval_t0 0 --eval_len $EL --tb none > $LG 2>&1 < /dev/null; }
CMPPY=$U/cmp_ck.py; cat > $CMPPY <<'PY_'
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False); b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
d = max(float((a["net"][k].float() - b["net"][k].float()).abs().max()) for k in a["net"] if k in b["net"])
s = max(float(b["net"][k].float().abs().max()) for k in b["net"])
oa = [t for st in a["opt"]["state"].values() for t in st.values() if torch.is_tensor(t) and t.numel() > 1]
ob = [t for st in b["opt"]["state"].values() for t in st.values() if torch.is_tensor(t) and t.numel() > 1]
o = max(float((x - y).abs().max()) for x, y in zip(oa, ob)) if oa and len(oa) == len(ob) else float("nan")
fz = sum("fuse" in k for k in a["net"])
print(f"반복 {a['k']}/{b['k']}  입력폭 {a['n_feat']}/{b['n_feat']}  키 같음 {sorted(a['net']) == sorted(b['net'])}  스킵 결합 키 {fz}  "
      f"망 파라미터 최대차 {d:.3e} (최대 크기 {s:.3e})  Adam 상태 최대차 {o:.3e}  생성기 같음 {bool((a['gen'] == b['gen']).all())}")
PY_
maxd(){ python -c "import re,sys;m=re.search(r'망 파라미터 최대차 ([0-9.e+-]+)', sys.argv[1]);sys.exit(0 if m and float(m.group(1))<$2 else 1)" "$1"; }
NFW=33; [ $M = gunet ] && NFW=31     # 입력폭: 모멘트 29 + 바닥 dist 4 / near 2
# ---- (1) 기본 경로 대조: 같은 명령 1 반복 -- 이 커밋 vs 기준 실행 커밋 (다르면 멈춤); 같은 1 반복을 다궤적 경로(L000 하나)로 (참고) ----
if [ ! -f $U/base_equiv_$C.ok ]; then
  wait_gpu; echo "[기본 경로 대조] GPU$G $(date +%T)"
  BE=$U/base_equiv_$C; [ -e $BE ] && mv $BE ${BE}_prev_$(date +%H%M%S); mkdir -p $BE/new $BE/old $BE/multi
  [ -d $REFWT ] || fail "기준 워크트리 없음 $REFWT"
  fit $BE/new 1 1 $BE/new.log 1
  ( cd $REFWT && fit $BE/old 1 1 $BE/old.log 1 )
  env AF_STATE_FIT_MULTI=L000=$DUMP:1:8 AF_STATE_FIT_LOSS=ip AF_STATE_FIT_EVERY=1 AF_STATE_FIT_EVAL=100000 AF_STATE_FIT_CK=$BE/multi/ck.pt AF_STATE_FIT_TB=$BE/multi/tb \
    CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py --infer --data $W/one/L000 --hold_traj L000 --out $BE/multi/run --tag EV $B --iters 0 --eval_t0 0 --eval_len 1 --tb none > $BE/multi.log 2>&1 < /dev/null
  TB=$(cat $BE/new.log $BE/old.log | grep -a -c Traceback)
  L1=$(grep -a -m1 "\[상태 적합\] 1/1 " $BE/new.log | grep -o "손실 [0-9.e+-]*"); L0=$(grep -a -m1 "\[상태 적합\] 1/1 " $BE/old.log | grep -o "손실 [0-9.e+-]*")
  LM=$(grep -a -m1 "\[다궤적 적합\] 1/1 " $BE/multi.log | grep -o "손실 [0-9.e+-]*")
  CMPB=$(python $CMPPY $BE/new/ck.pt $BE/old/ck.pt 2>&1 | tail -1); CMPM=$(python $CMPPY $BE/new/ck.pt $BE/multi/ck.pt 2>&1 | tail -1)
  { echo "[기본 경로 대조 $M] 이 커밋 $C 첫 반복 $L1 | 기준 커밋 $REFC 첫 반복 $L0 | $CMPB | Traceback $TB"
    echo "[참고: 같은 1 반복을 다궤적 경로(AF_STATE_FIT_MULTI, L000 하나)로] 첫 반복 $LM | 한 궤적 경로 대비: $CMPM | Traceback $(grep -a -c Traceback $BE/multi.log)"; } | tee $U/base_equiv.txt
  [ -n "$L1" ] && [ "$L1" = "$L0" ] && grep -q "입력폭 $NFW/$NFW  키 같음 True" <<<"$CMPB" && [ "$TB" = 0 ] || fail "기본 경로가 기준 커밋 $REFC 와 다르다 (손실·입력폭·키)"
  maxd "$CMPB" 1e-5 || fail "기본 경로 망 파라미터 차이 1e-5 이상"
  touch $U/base_equiv_$C.ok; up
fi
# ---- (2) 짧은 시험 겸 이어 돌리기 시험: A = 200 반복(100 마다 저장·검증 80 프레임) -> 같은 체크포인트로 300, B = 한 번에 300 ----
if [ ! -f $U/resume_$C.ok ]; then
  wait_gpu; echo "[짧은 시험·이어 돌리기 시험] GPU$G $(date +%T)"
  RT=$U/resume_$C/test; RB=$U/resume_$C/ref; [ -e $U/resume_$C ] && mv $U/resume_$C $U/resume_${C}_prev_$(date +%H%M%S); mkdir -p $RT $RB
  t0=$(date +%s); fit $RT 200 100 $RT/r200.log 80; t1=$(date +%s); fit $RT 300 100 $RT/r300.log 80; t2=$(date +%s)
  fit $RB 300 100 $RB/r300.log 80; t3=$(date +%s)
  K=$(ck_k $RT/ck.pt); RS=$(grep -a -m1 "이어 돈다" $RT/r300.log | cut -c1-120)
  NF=$(grep -a -m1 "새 망, 입력 표준화\|Graph U-Net\] 파라미터" $RT/r200.log | cut -c1-200)
  TB=$(cat $RT/r200.log $RT/r300.log $RB/r300.log | grep -a -c Traceback)
  CMP=$(python $CMPPY $RT/ck.pt $RB/ck.pt 2>&1 | tail -1)
  EV=$(grep -a "상태 적합 롤아웃\]" $RB/r300.log | cut -c1-120 | tr '\n' ' ')
  LS=$(grep -a "\[상태 적합\] [0-9]*/300 " $RB/r300.log | sed 's/.*\] //' | cut -c1-60 | tr '\n' ' ')
  { echo "[이어 돌리기 $M] 300 실행: $RS | 마지막 반복 $K | Traceback $TB | 이어 돈 것 vs 한 번에: $CMP"
    echo "[시간] 200 반복 $((t1-t0)) s (검증 2 번 포함), 이어 100 반복 $((t2-t1)) s, 한 번에 300 반복 $((t3-t2)) s (검증 3 번 포함) | $NF"
    echo "[짧은 시험 300 반복] 손실: $LS"
    echo "[짧은 시험 검증 (정답 시작 80 프레임)] $EV"; } | tee $U/resume_test.txt
  [ "$K" = 300 ] && [ -n "$RS" ] && [ "$TB" = 0 ] && grep -q "입력폭 $NFW/$NFW " <<<"$CMP" && grep -q "생성기 같음 True" <<<"$CMP" || fail "이어 돌리기 확인 실패 -- 학습 안 함"
  [ $M = gunet ] && grep -q "스킵 결합 키 0 " <<<"$CMP" && fail "Graph U-Net 스킵 결합 키 없음"
  maxd "$CMP" 1e-3 || fail "이어 돈 것과 한 번에 돈 것의 망 차이가 1e-3 이상"
  for T in L000 0004; do
    env AF_INV_LOG=$RT/inv_$T.jsonl AF_TUNNEL_LOG=$RT/tun_$T.jsonl AF_ROLL_DUMP=$RT/roll_$T.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$G \
      python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $RT/roll_$T --tag EV --resume $RT/ck.pt $B --iters 0 --eval_t0 0 --eval_len 3 --tb none > $RT/roll_$T.log 2>&1 < /dev/null
    RL=$(grep -a "롤아웃\] $T" $RT/roll_$T.log | tail -1 | cut -c1-160); echo "[롤아웃 연기 시험 $T] $RL | Traceback $(grep -a -c Traceback $RT/roll_$T.log) | 덤프 $(ls -la $RT/roll_$T.pt 2>/dev/null | awk '{print $5}') B | 뒤집힘 기록 $(wc -l < $RT/inv_$T.jsonl 2>/dev/null) 줄" | tee -a $U/resume_test.txt
    [ -n "$RL" ] && [ "$(grep -a -c Traceback $RT/roll_$T.log)" = 0 ] && [ -f $RT/roll_$T.pt ] || fail "롤아웃 연기 시험 실패 ($T) -- 학습 안 함"
  done
  touch $U/resume_$C.ok; up
fi
if [ -n "${SF1_TEST_ONLY:-}" ]; then touch $U/SF1_${MU}_TEST_DONE; up; echo "SF1_${MU}_TEST_DONE"; exit 0; fi
# ---- (3) 학습 15000 반복 (500 마다 체크포인트 -> R2 바로, 같은 때 정답 시작 80 프레임 검증) ----
mkdir -p $U/sf
K=$(ck_k $U/sf/ck.pt 2>/dev/null)
if [ "$K" != $NIT ]; then
  wait_gpu
  CMD="env AF_STATE_FIT=$DUMP:$NIT:8 AF_STATE_FIT_LOSS=ip AF_STATE_FIT_CK=$U/sf/ck.pt AF_STATE_FIT_R2=$R/sf/ck.pt AF_STATE_FIT_TB=$U/sf/tb CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py --infer --data $W/one/L000 --hold_traj L000 --out $U/sf/run --tag EV $B --iters 0 --eval_t0 0 --eval_len 80 --tb none"
  echo "[학습 시작] sf1 $M: 커밋 $C, GPU$G, 이어 돌기 체크포인트 반복 ${K:-없음}, $(date '+%F %T')" | tee -a $U/SF1_LAUNCHED
  echo "[학습 명령줄] $CMD" | tee -a $U/SF1_LAUNCHED; up
  $CMD >> $U/sf/run.log 2>&1 < /dev/null
  grep -a "상태 적합 롤아웃\|rror" $U/sf/run.log | tail -4 | cut -c1-200
  K=$(ck_k $U/sf/ck.pt); echo "[학습 끝] 반복 $K $(date +%T)"; up
  [ "$K" = $NIT ] || fail "$NIT 못 미침 (반복 $K) -- 롤아웃 생략"
fi
# ---- (4) L000·0004 롤아웃 (F float32 덤프) + 3DGS 영상 + psi 색 (같은 GPU 에서 차례로; 다섯 궤적 두 실행과 같은 명령·도구) ----
roll(){ T=$1; N=sf1_${M}_$T
  if [ ! -f $U/roll_$N.pt ] || ! grep -a -q "롤아웃\] $T" $U/v_$N.log 2>/dev/null; then
    for f in $U/roll_$N.pt $U/inv_$N.jsonl $U/tun_$N.jsonl $U/v_$N.log; do [ -e $f ] && mv $f $f.prev_$(date +%H%M%S); done
    echo "[롤아웃 명령] AF_INV_LOG=$U/inv_$N.jsonl AF_TUNNEL_LOG=$U/tun_$N.jsonl AF_ROLL_DUMP=$U/roll_$N.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $U/v_$N --tag EV --resume $U/sf/ck.pt $B --iters 0 --eval_t0 0 --eval_len 80 --tb none" > $U/cmd_roll_$T.txt
    env AF_INV_LOG=$U/inv_$N.jsonl AF_TUNNEL_LOG=$U/tun_$N.jsonl AF_ROLL_DUMP=$U/roll_$N.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$G \
      python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $U/v_$N --tag EV --resume $U/sf/ck.pt $B --iters 0 --eval_t0 0 --eval_len 80 --tb none > $U/v_$N.log 2>&1 < /dev/null
    bgup $U/roll_$N.pt; fi
  grep -a "롤아웃\] $T\|Traceback\|Error" $U/v_$N.log | tail -2 | cut -c1-200
  [ -f $U/roll_$N.pt ] || { echo "[오류] $T 롤아웃 덤프 없음"; return 1; }
  echo "$N $(grep -a '롤아웃\] ' $U/v_$N.log | tail -1 | grep -o '평균 [0-9.]*%')" > $U/err_$N.txt; up
  [ -f $U/h5_$N/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $U/roll_$N.pt --traj $(gt_of $T) --out $U/h5_$N > $U/h5_$N.log 2>&1
  [ -f $U/$N.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $U/h5_$N --config $W/lbdata/config.json --shape lego --out $U/$N.mp4 > $U/r_$N.log 2>&1 < /dev/null
  [ -f $U/$N.mp4 ] || { echo "[오류] $T 렌더"; return 1; }
  up
  # psi: psi_color_lb (d993fb5 워크트리, noinertial·gunet5_near 와 같은 도구·인자), 공통 고정 범위; 칸 $CN = 이번
  D=$U/psi/$T; mkdir -p $D
  ( cd $WP && { [ -f $D/col_$CN.npy ] || CUDA_VISIBLE_DEVICES=$G python -u exe/psi_color_lb.py --gt_dir $(gt_of $T) --pt $(pt_of $T) --dump $CN=$U/roll_$N.pt --out $D \
        --eps 1e-3 --q 1 99 --cmap turbo --range $RANGE --stats_name defstats_$T.txt --stats_frames 1 40 80 > $D/psi_color.log 2>&1 < /dev/null; }
    grep -q PSI_COLOR_DONE $D/psi_color.log || { tail -5 $D/psi_color.log; exit 1; }
    [ -f $D/h5_$CN/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $U/roll_$N.pt --traj $(gt_of $T) --out $D/h5_$CN > $D/h5_$CN.log 2>&1
    [ -f $D/psi_${CN}_lego.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $D/h5_$CN --config $W/lbdata/config.json --shape lego \
       --colors $D/col_$CN.npy --out $D/psi_${CN}_lego.mp4 > $D/r_$CN.log 2>&1 < /dev/null
    [ -f $D/psi_${CN}_lego.mp4 ] ) || { echo "[오류] $T psi"; return 1; }
  grep -a "\[색 범위\]" $D/psi_color.log; cat $D/defstats_$T.txt
  # 정답 칸은 같은 psi·같은 범위로 이미 만든 렌더를 재사용 (noinertial psi/<T>); 범위 대조용 range 도 복사
  cp $NI/psi/$T/psi_gt_lego.mp4 $D/psi_gt_lego.mp4; cp $NI/psi/$T/range.json $D/range_noinertial.json
  touch $U/.done_roll_$T; }
for T in L000 0004; do
  [ -f $U/.done_roll_$T ] || { wait_gpu; roll $T > $U/task_roll_$T.log 2>&1; cat $U/task_roll_$T.log | cut -c1-200; }
  [ -f $U/.done_roll_$T ] || fail "롤아웃·영상·psi 단계 실패 ($T)"
done
cat $U/err_sf1_${M}_L000.txt $U/err_sf1_${M}_0004.txt > $U/errors.txt
# ---- (5) 학습 곡선 (TB -> json): 이번(한 궤적) + 다섯 궤적 두 실행 ----
python - $U/sf/tb $NI/multi/tb $G5/multi/tb $U/curves.json $M <<'PY'
import json, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
out = {}
for nm, d in (("sf1_" + sys.argv[5], sys.argv[1]), ("noinertial", sys.argv[2]), ("gunet5_near", sys.argv[3])):
    ea = EventAccumulator(d, size_guidance={"scalars": 0}); ea.Reload()
    out[nm] = {k: [(e.step, e.value) for e in ea.Scalars(k)] for k in ea.Tags()["scalars"]
               if k.startswith("상태적합/") or k.startswith("상태적합_롤아웃/") or k.startswith("다궤적적합/") or k.startswith("다궤적적합_롤아웃_")}
json.dump(out, open(sys.argv[4], "w"), ensure_ascii=False)
print("곡선", {k: len(v) for k, v in out.items()})
PY
up
kill $SYNC 2>/dev/null
rclone copy $U $R -q && echo "[R2 업로드 완료] $(date)" && touch $U/ALL_DONE $U/SF1_${MU}_CLUSTER_DONE && rclone copy $U $R -q
rclone check $U $R --one-way --size-only -q && echo "[R2 대조] 전부 올라감"
echo SF1_${MU}_CLUSTER_DONE
