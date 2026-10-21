#!/bin/bash
# lego 새 바닥 다섯 궤적 상태 적합 -- 노드 입력을 입자 물리량 주파수 인코딩으로 (--pfreq K, 사용자 10-21)
#   설정 = spstick.sh multi 글자 그대로 (BASE 를 spstick.sh 원문에서 다시 만들어 대조, spstick.out [BASE] 줄과도 대조) + 플래그 하나:
#     --pfreq K (기본 6, 환경변수 PFREQ_K): 입자 모멘트 29 채널을 빼고 15(1+2K) 채널 (바닥 4 채널은 그대로) -> 입력 15(1+2K)+4
# 순서: 단위 시험(CPU) -> 기본 경로 대조(플래그 없이 1 반복: 이 커밋 vs spstick 커밋 13471f4, 기록만)
#       -> 이어 돌리기 시험(20 -> 30 이어 vs 한 번에 30, 망·Adam 비교) -> 롤아웃 연기 시험 -> 15000 반복
#       -> L000·0004 롤아웃 (F float32 덤프) + 3DGS 영상 + psi 색 (공통 고정 범위 lbours/lego_psi/range.json) -> 학습 곡선 json
# 멱등: 잡이 바뀌어 다시 띄우면 끝난 단계는 건너뛰고 학습은 체크포인트(500 마다, R2 즉시)에서 이어 돈다.
# 출력은 처음부터 R2 lbours/pfreq 와 통째 동기화 (제외 없음, 5분 루프). 잡 안에서만: AF_DETACH=1 bash exe/job_run.sh $W/pfreq_boot.sh <로그>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD); KF=${PFREQ_K:-6}
P=$W/lbpool; U=$W/lbours/pfreq; S=$W/lbours/spstick; NS=$W/lbours/lego_psi_spstick; WS=$W/wt/spstick_13471f4
WP=$W/wt/psi_d993fb5; RANGE=$W/lbours/lego_psi/range.json
R=r2:storage/result/anchorflow/lbours/pfreq; mkdir -p $U
if [ -f $U/driver.hb ] && [ $(( $(date +%s) - $(cat $U/driver.hb) )) -lt 300 ]; then echo "[중단] 다른 드라이버가 돌고 있다 (하트비트 5 분 안) $(date)"; exit 0; fi
echo $$ > $U/driver.pid
( while true; do date +%s > $U/driver.hb; sleep 120; done ) & HB=$!
( while true; do rclone copy $U $R -q; sleep 300; done ) & SYNC=$!
trap 'kill $SYNC $HB 2>/dev/null' EXIT
echo "[시작] $(date) 커밋 $C 노드 $(hostname) 잡 ${PBS_JOBID:-?} K=$KF"
up(){ rclone copy $U $R -q --include "*.mp4" --include "*.txt" --include "*.json" --include "*.jsonl" --include "*.log" --include "PFREQ_*" --max-depth 1; }
bgup(){ ( rclone copyto $1 $R/${1#$U/} -q && echo "[R2 덤프 올림] ${1#$U/} $(date +%T)" ) & }
fail(){ echo "[오류] $1 $(date)"; echo "$1" > $U/PFREQ_FAIL; up; exit 1; }
# ---- spstick.sh 와 같은 식 (그대로 옮김) ----
COMMON="--no_mat --arch sgnn --gnn_layers 8 --hidden 128 --obj pts --net_inertial"
REST_OLD="--inv_barrier 1.0 --inv_jhat 0.3 --inv_ext linear --node_h 0.028284 --dt_cond --dt_scale --v0_traj --det_eps 0.1 --det_w 100 --lr 3e-4 --batch 8 --det_every 200 --pool_fill $P/fill --pool_cfg $P/cfg --pool_combos lego_jelly --pool_drop 0.0,0.02,0.0,1.5 --pool_frames 80 --phase2 --phys_w 1.0 --save_every 100000"
FLOOR_SP="--bc_level node --bc_mode project --bc_softplus 0.01"
REST=$(echo "$REST_OLD" | sed -e 's/--inv_barrier 1.0 --inv_jhat 0.3 --inv_ext linear/--inv_barrier 0/' -e 's/--det_w 100/--det_w 0 --det_keep_elastic/')
BASE_SP="$COMMON $FLOOR_SP $REST"
BASE_NOXN="$BASE_SP --bc_stick 0.01"
BASE="$BASE_NOXN --bc_stick_xn"
BASE_PF="$BASE --pfreq $KF"
SPB=$(P=$P bash -c "$(grep -E '^(COMMON|REST_OLD|FLOOR_SP|REST|BASE_SP|BASE_NOXN|BASE)=' $W/spstick.sh; echo 'echo "$BASE"')")
OUTD=$(comm -23 <(tr ' ' '\n' <<<"$SPB" | sort) <(tr ' ' '\n' <<<"$BASE_PF" | sort) | paste -sd' ')
IND=$(comm -13 <(tr ' ' '\n' <<<"$SPB" | sort) <(tr ' ' '\n' <<<"$BASE_PF" | sort) | paste -sd' ')
{ echo "[spstick BASE] $SPB"; echo "[pfreq BASE  ] $BASE_PF"
  echo "[명령줄 차이 spstick multi -> pfreq] 플래그 빠짐: '$OUTD' | 플래그 더함: '$IND'"; } | tee $U/cmdline_diff.txt
[ "$SPB" = "$BASE" ] || fail "BASE 가 spstick.sh 와 글자 그대로가 아니다"
[ "$BASE_PF" = "$SPB --pfreq $KF" ] && [ -z "$OUTD" ] || fail "명령줄 차이가 --pfreq $KF 하나가 아니다 ('$OUTD' / '$IND')"
echo "[대조] pfreq BASE == spstick BASE + ' --pfreq $KF' (문자열 동일)" | tee -a $U/cmdline_diff.txt
[ "[BASE] $BASE" = "$(grep -a -m1 '^\[BASE\]' $W/spstick.out)" ] && echo "[BASE] spstick 학습 실행 로그(spstick.out)와 일치" | tee -a $U/cmdline_diff.txt || fail "BASE 가 spstick.out 과 다르다"
gt_of(){ [ $1 = L000 ] && echo $W/lbdata/test_low/L000 || echo $W/lbdata/test/$1; }
pt_of(){ [ $1 = L000 ] && echo $W/lbpt/test_low/L000.pt || echo $W/lbpt/test/$1.pt; }
cp -n $W/lbdata/test/0004/video.mp4 $U/gt_0004.mp4; cp -n $W/lbdata/test_low/L000/video.mp4 $U/gt_L000.mp4
ck_k(){ python -c "import torch;c=torch.load('$1',map_location='cpu',weights_only=False);print(c.get('k',c.get('it',c.get('step','?'))))" 2>/dev/null; }
SPEC=""; for T in L000 0000 0001 0002 0003; do SPEC="$SPEC${SPEC:+,}$T=$S/roll_ov_$T.pt"; done
for T in L000 0000 0001 0002 0003; do [ -f $S/roll_ov_$T.pt ] || fail "학습 상태 덤프 없음 $S/roll_ov_$T.pt"; done
free_gpu(){ # 빈 GPU (메모리 1GB 미만 + 프로세스 없음) 하나, 인자 카드는 빼고
  for c in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<1024{print $1}'); do
    [ "$c" = "${1:-x}" ] && continue
    u=$(nvidia-smi -i $c --query-gpu=uuid --format=csv,noheader)
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u" && continue
    echo $c; return; done; }
wait_gpu(){ local g=""; while [ -z "$g" ]; do g=$(free_gpu "$1"); [ -z "$g" ] && { echo "[대기] 빈 GPU 없음 $(date +%T)" >> $U/wait.log; sleep 120; }; done
  echo "[GPU 배정] $g $(date '+%F %T') $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $U/gpu.log; echo $g; }
fit(){ # fit <출력디렉토리> <반복> <저장간격> <gpu> <로그> <BASE...>
  local O=$1 N=$2 E=$3 G=$4 LG=$5; shift 5
  env AF_STATE_FIT_MULTI=$SPEC:$N:8 AF_STATE_FIT_LOSS=ip AF_STATE_FIT_EVERY=$E AF_STATE_FIT_EVAL=100000 AF_STATE_FIT_CK=$O/ck.pt AF_STATE_FIT_TB=$O/tb \
    CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py --infer --data $W/multi5 --hold_traj L000,0000,0001,0002,0003 --out $O/run --tag EV "$@" --iters 0 --eval_t0 0 --eval_len 1 --tb none > $LG 2>&1 < /dev/null; }
# ---- 0) 단위 시험 (CPU) ----
if [ ! -f $U/unit_test_$C.ok ]; then
  CUDA_VISIBLE_DEVICES= python -u exe/test_pfreq.py > $U/unit_test.log 2>&1 < /dev/null; { cat $U/unit_test.log; echo "커밋 $C"; } | tee $U/unit_test.txt
  grep -q PFREQ_TEST_OK $U/unit_test.log || fail "단위 시험 실패 -- 실행 안 함"
  touch $U/unit_test_$C.ok; up
fi
# ---- 0b) 기본 경로 대조 (기록만): 플래그 없이 같은 BASE 로 1 반복 -- 이 커밋 vs spstick 학습 커밋 13471f4 ----
if [ ! -f $U/base_equiv_$C.ok ]; then
  g=$(wait_gpu); echo "[기본 경로 대조] GPU$g $(date +%T)"
  BE=$U/base_equiv_$C; [ -e $BE ] && mv $BE ${BE}_prev_$(date +%H%M%S); mkdir -p $BE/new $BE/old
  fit $BE/new 1 1 $g $BE/new.log $BASE
  ( cd $WS && fit $BE/old 1 1 $g $BE/old.log $BASE )
  CMPB=$(python - $BE/new/ck.pt $BE/old/ck.pt <<'PY'
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False); b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
st = {k: float((a["net"][k].float() - b["net"][k].float()).abs().max()) for k in a["net"] if k in b["net"]}
mu = {k: v for k, v in st.items() if "in_mu" in k or "in_sd" in k}
pr = {k: v for k, v in st.items() if k not in mu}
print(f"입력폭 {a.get('n_feat')}/{b.get('n_feat')}  키 같음 {sorted(a['net']) == sorted(b['net'])}  표준화 통계 최대차 {max(mu.values()) if mu else float('nan'):.3e}  "
      f"1 반복 뒤 망 파라미터 최대차 {max(pr.values()):.3e}")
PY
)
  L1=$(grep -a -m1 "\[다궤적 적합\] 1/1 " $BE/new.log | grep -o "손실 [0-9.e+-]*"); L0=$(grep -a -m1 "\[다궤적 적합\] 1/1 " $BE/old.log | grep -o "손실 [0-9.e+-]*")
  echo "[기본 경로 대조] 이 커밋 $C 첫 반복 $L1 | 13471f4 첫 반복 $L0 | $CMPB | Traceback $(cat $BE/new.log $BE/old.log | grep -a -c Traceback)" | tee $U/base_equiv.txt
  touch $U/base_equiv_$C.ok; up
fi
# ---- 1) 이어 돌리기 시험: A = 20 반복(10 마다 저장) 뒤 같은 체크포인트로 30 까지, B = 한 번에 30 -> 비교 ----
if [ ! -f $U/multi_resume_$C.ok ]; then
  g=$(wait_gpu); echo "[이어 돌리기 시험] GPU$g $(date +%T)"
  RT=$U/resume_$C/test; RB=$U/resume_$C/ref; [ -e $U/resume_$C ] && mv $U/resume_$C $U/resume_${C}_prev_$(date +%H%M%S); mkdir -p $RT $RB
  t0=$(date +%s); fit $RT 20 10 $g $RT/r20.log $BASE_PF; t1=$(date +%s); fit $RT 30 10 $g $RT/r30.log $BASE_PF; t2=$(date +%s)
  fit $RB 30 10 $g $RB/r30.log $BASE_PF
  K=$(ck_k $RT/ck.pt); RS=$(grep -a -m1 "이어 돈다" $RT/r30.log | cut -c1-120); NF=$(grep -a -m1 "\[--pfreq" $RT/r20.log | cut -c1-200)
  TB=$(cat $RT/r20.log $RT/r30.log $RB/r30.log | grep -a -c Traceback)
  CMP=$(python - $RT/ck.pt $RB/ck.pt <<'PY'
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False); b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
d = max(float((a["net"][k].float() - b["net"][k].float()).abs().max()) for k in a["net"])
s = max(float(b["net"][k].float().abs().max()) for k in b["net"])
o = max(float((x - y).abs().max()) for x, y in zip([t for st in a["opt"]["state"].values() for t in st.values() if torch.is_tensor(t) and t.numel() > 1],
                                                    [t for st in b["opt"]["state"].values() for t in st.values() if torch.is_tensor(t) and t.numel() > 1]))
print(f"반복 {a['k']}/{b['k']}  입력폭 {a['n_feat']}  망 파라미터 최대차 {d:.3e} (최대 크기 {s:.3e})  Adam 상태 최대차 {o:.3e}  생성기 같음 {bool((a['gen'] == b['gen']).all())}")
PY
)
  echo "[이어 돌리기 multi] 30 실행: $RS | 마지막 반복 $K | Traceback $TB | 이어 돈 것 vs 한 번에: $CMP | 20 반복 $((t1-t0)) s, 이어 10 반복 $((t2-t1)) s | 입력: $NF" | tee $U/resume_test.txt
  NFX=$(( 15 * (1 + 2 * KF) + 4 ))
  [ "$K" = 30 ] && [ -n "$RS" ] && [ "$TB" = 0 ] && grep -q "입력폭 $NFX " <<<"$CMP" && grep -q "생성기 같음 True" <<<"$CMP" || fail "이어 돌리기 확인 실패 -- 학습 안 함"
  python -c "import re,sys;m=re.search(r'망 파라미터 최대차 ([0-9.e+-]+)', sys.argv[1]);sys.exit(0 if m and float(m.group(1))<1e-3 else 1)" "$CMP" || fail "이어 돈 것과 한 번에 돈 것의 망 차이가 1e-3 이상"
  env AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$g python -u exe/train_deform.py --infer --data $W/one/L000 --hold_traj L000 --out $RT/roll --tag EV --resume $RT/ck.pt $BASE_PF --iters 0 --eval_t0 0 --eval_len 3 --tb none > $RT/roll.log 2>&1 < /dev/null
  RL=$(grep -a "롤아웃\] L000" $RT/roll.log | tail -1 | cut -c1-160); echo "[롤아웃 연기 시험] $RL | Traceback $(grep -a -c Traceback $RT/roll.log)" | tee -a $U/resume_test.txt
  [ -n "$RL" ] && [ "$(grep -a -c Traceback $RT/roll.log)" = 0 ] || fail "롤아웃 연기 시험 실패 -- 학습 안 함"
  touch $U/multi_resume_$C.ok; up
fi
# ---- 2) 학습 15000 반복 (500 마다 체크포인트 -> R2 바로, 1000 마다 다섯 궤적 검증) ----
mkdir -p $U/multi
K=$(ck_k $U/multi/ck.pt 2>/dev/null)
if [ "$K" != 15000 ]; then
  g=$(wait_gpu)
  CMD="env AF_STATE_FIT_MULTI=$SPEC:15000:8 AF_STATE_FIT_LOSS=ip AF_STATE_FIT_CK=$U/multi/ck.pt AF_STATE_FIT_R2=$R/multi/ck.pt AF_STATE_FIT_TB=$U/multi/tb CUDA_VISIBLE_DEVICES=$g python -u exe/train_deform.py --infer --data $W/multi5 --hold_traj L000,0000,0001,0002,0003 --out $U/multi/run --tag EV $BASE_PF --iters 0 --eval_t0 0 --eval_len 80 --tb none"
  echo "[학습 시작] pfreq multi: 커밋 $C, --pfreq $KF, GPU$g, 이어 돌기 체크포인트 반복 ${K:-없음}, $(date '+%F %T')" | tee -a $U/PFREQ_LAUNCHED
  echo "[학습 명령줄] $CMD" | tee -a $U/PFREQ_LAUNCHED; up
  $CMD >> $U/multi/run.log 2>&1 < /dev/null
  grep -a "다궤적 롤아웃\|rror" $U/multi/run.log | tail -8 | cut -c1-200
  K=$(ck_k $U/multi/ck.pt); echo "[multi 학습 끝] 반복 $K $(date +%T)"; up
  [ "$K" = 15000 ] || fail "multi 15000 못 미침 (반복 $K) -- 롤아웃 생략"
fi
# ---- 3) L000·0004 롤아웃 (F float32 덤프) + 3DGS 영상 + psi 색 (GPU 둘 동시에) ----
roll(){ T=$1; g=$2; N=pfreq_multi_$T
  if [ ! -f $U/roll_$N.pt ] || ! grep -a -q "롤아웃\] $T" $U/v_$N.log 2>/dev/null; then rm -f $U/roll_$N.pt $U/inv_$N.jsonl $U/tun_$N.jsonl
    echo "[롤아웃 명령] AF_INV_LOG=$U/inv_$N.jsonl AF_TUNNEL_LOG=$U/tun_$N.jsonl AF_ROLL_DUMP=$U/roll_$N.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$g python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $U/v_$N --tag EV --resume $U/multi/ck.pt $BASE_PF --iters 0 --eval_t0 0 --eval_len 80 --tb none" > $U/cmd_roll_$T.txt
    env AF_INV_LOG=$U/inv_$N.jsonl AF_TUNNEL_LOG=$U/tun_$N.jsonl AF_ROLL_DUMP=$U/roll_$N.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$g \
      python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $U/v_$N --tag EV --resume $U/multi/ck.pt $BASE_PF --iters 0 --eval_t0 0 --eval_len 80 --tb none > $U/v_$N.log 2>&1 < /dev/null
    bgup $U/roll_$N.pt; fi
  grep -a "롤아웃\] $T\|Traceback\|Error" $U/v_$N.log | tail -2 | cut -c1-200
  [ -f $U/roll_$N.pt ] || { echo "[오류] $T 롤아웃 덤프 없음"; return 1; }
  echo "$N $(grep -a '롤아웃\] ' $U/v_$N.log | tail -1 | grep -o '평균 [0-9.]*%')" > $U/err_$N.txt; up
  [ -f $U/h5_$N/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $U/roll_$N.pt --traj $(gt_of $T) --out $U/h5_$N > $U/h5_$N.log 2>&1
  [ -f $U/$N.mp4 ] || CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $U/h5_$N --config $W/lbdata/config.json --shape lego --out $U/$N.mp4 > $U/r_$N.log 2>&1 < /dev/null
  [ -f $U/$N.mp4 ] || { echo "[오류] $T 렌더"; return 1; }
  up
  # psi: psi_color_lb (d993fb5 워크트리, lego_psi_spstick·relexcess 와 같은 도구·인자), 공통 고정 범위
  D=$U/psi/$T; mkdir -p $D
  ( cd $WP && { [ -f $D/col_pf.npy ] || CUDA_VISIBLE_DEVICES=$g python -u exe/psi_color_lb.py --gt_dir $(gt_of $T) --pt $(pt_of $T) --dump mm=$NS/roll_mm_$T.pt --dump pf=$U/roll_$N.pt --out $D \
        --eps 1e-3 --q 1 99 --cmap turbo --range $RANGE --stats_name defstats_$T.txt --stats_frames 1 40 80 > $D/psi_color.log 2>&1 < /dev/null; }
    grep -q PSI_COLOR_DONE $D/psi_color.log || { tail -5 $D/psi_color.log; exit 1; }
    rm -f $D/col_gt.npy $D/col_mm.npy   # 정답·기존 칸은 lego_psi_spstick 의 같은 범위 렌더를 재사용 (같은 psi·같은 범위로 만든 색)
    [ -f $D/h5_pf/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $U/roll_$N.pt --traj $(gt_of $T) --out $D/h5_pf > $D/h5_pf.log 2>&1
    [ -f $D/psi_pf_lego.mp4 ] || CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $D/h5_pf --config $W/lbdata/config.json --shape lego \
       --colors $D/col_pf.npy --out $D/psi_pf_lego.mp4 > $D/r_pf.log 2>&1 < /dev/null
    [ -f $D/psi_pf_lego.mp4 ] ) || { echo "[오류] $T psi"; return 1; }
  grep -a "\[색 범위\]" $D/psi_color.log; cat $D/defstats_$T.txt
  cp $NS/$T/psi_gt_lego.mp4 $D/psi_gt_lego.mp4; cp $NS/$T/psi_mm_lego.mp4 $D/psi_mm_lego.mp4; cp $NS/$T/range.json $D/range_spstick.json
  touch $U/.done_roll_$T; }
if [ ! -f $U/.done_roll_L000 ] || [ ! -f $U/.done_roll_0004 ]; then
  g1=$(wait_gpu); ( [ -f $U/.done_roll_L000 ] || roll L000 $g1 > $U/task_roll_L000.log 2>&1 ) & P1=$!; sleep 90
  g2=$(wait_gpu $g1); ( [ -f $U/.done_roll_0004 ] || roll 0004 $g2 > $U/task_roll_0004.log 2>&1 ) & P2=$!
  wait $P1 $P2
  cat $U/task_roll_L000.log $U/task_roll_0004.log | cut -c1-200
  [ -f $U/.done_roll_L000 ] && [ -f $U/.done_roll_0004 ] || fail "롤아웃·영상·psi 단계 실패"
fi
grep -h "재롤아웃" $NS/job_L000.out $NS/job_0004.out > $U/err_spstick_mm.txt
cat $U/err_pfreq_multi_L000.txt $U/err_pfreq_multi_0004.txt > $U/errors_multi.tmp
# ---- 4) 학습 곡선 (TB -> json): 이번·기존 spstick multi ----
python - $U/multi/tb $S/multi/tb $U/curves.json <<'PY'
import json, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
out = {}
for nm, d in (("pfreq", sys.argv[1]), ("spstick", sys.argv[2])):
    ea = EventAccumulator(d, size_guidance={"scalars": 0}); ea.Reload()
    out[nm] = {k: [(e.step, e.value) for e in ea.Scalars(k)] for k in ea.Tags()["scalars"]
               if k.startswith("다궤적적합/") or k.startswith("다궤적적합_롤아웃_")}
json.dump(out, open(sys.argv[3], "w"), ensure_ascii=False)
print("곡선", {k: len(v) for k, v in out.items()})
PY
mv $U/errors_multi.tmp $U/errors_multi.txt; up
kill $SYNC 2>/dev/null
rclone copy $U $R -q && echo "[R2 업로드 완료] $(date)" && touch $U/ALL_DONE $U/PFREQ_CLUSTER_DONE && rclone copy $U $R -q
rclone check $U $R --one-way --size-only -q && echo "[R2 대조] 전부 올라감"
echo PFREQ_CLUSTER_DONE
