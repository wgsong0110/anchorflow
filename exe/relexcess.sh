#!/bin/bash
# lego 새 바닥 다섯 궤적 상태 적합 -- 상태별 정규화 손실 + 공중/접촉 균형 배치 (사용자 10-21, 진단 ovnet/lego_grad_air_vs_contact.md)
#   설정 = spstick.sh multi 글자 그대로 (BASE 를 spstick.sh 원문에서 다시 만들어 대조, 학습 로그 [BASE] 줄과도 대조) + 환경변수 둘:
#     AF_SF_RELEX=1      손실 = (E_s − E_s*) / D_s + det 벌점 (exe/train_deform.py 의 AF_SF_RELEX 블록)
#     AF_SF_BALANCE=4:4  배치 8 = 공중 4 + 접촉 4 (상태 시작 높이 > 20ε = 0.2)
# 순서: 단위 시험(새 망·학습 끝 망) -> 이어 돌리기 시험(20 -> 30 이어 vs 한 번에 30) -> 롤아웃 연기 시험 -> 15000 반복
#       -> L000·0004 롤아웃 (F float32 덤프) + 3DGS 영상 -> psi 색 (공통 고정 범위 lbours/lego_psi/range.json) -> 학습 곡선 json
# 멱등: 잡이 바뀌어 다시 띄우면 끝난 단계는 건너뛰고 학습은 체크포인트(500 마다, R2 즉시)에서 이어 돈다.
# 출력은 처음부터 R2 lbours/relexcess 와 통째 동기화 (제외 없음, 5분 루프). 잡 안에서만: AF_DETACH=1 bash exe/job_run.sh $W/relexcess_boot.sh <로그>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
P=$W/lbpool; U=$W/lbours/relexcess; S=$W/lbours/spstick; GA=$W/lbours/grad_air_contact; NS=$W/lbours/lego_psi_spstick
WP=$W/wt/psi_d993fb5; RANGE=$W/lbours/lego_psi/range.json
R=r2:storage/result/anchorflow/lbours/relexcess; mkdir -p $U
if [ -f $U/driver.hb ] && [ $(( $(date +%s) - $(cat $U/driver.hb) )) -lt 300 ]; then echo "[중단] 다른 드라이버가 돌고 있다 (하트비트 5 분 안) $(date)"; exit 0; fi
echo $$ > $U/driver.pid
( while true; do date +%s > $U/driver.hb; sleep 120; done ) & HB=$!
( while true; do rclone copy $U $R -q; sleep 300; done ) & SYNC=$!
trap 'kill $SYNC $HB 2>/dev/null' EXIT
echo "[시작] $(date) 커밋 $C 노드 $(hostname) 잡 ${PBS_JOBID:-?}"
up(){ rclone copy $U $R -q --include "*.mp4" --include "*.txt" --include "*.json" --include "*.jsonl" --include "*.log" --include "RELEXCESS_*" --max-depth 1; }
bgup(){ ( rclone copyto $1 $R/${1#$U/} -q && echo "[R2 덤프 올림] ${1#$U/} $(date +%T)" ) & }
fail(){ echo "[오류] $1 $(date)"; echo "$1" > $U/RELEXCESS_FAIL; up; exit 1; }
# ---- spstick.sh 와 같은 식 (그대로 옮김) ----
COMMON="--no_mat --arch sgnn --gnn_layers 8 --hidden 128 --obj pts --net_inertial"
REST_OLD="--inv_barrier 1.0 --inv_jhat 0.3 --inv_ext linear --node_h 0.028284 --dt_cond --dt_scale --v0_traj --det_eps 0.1 --det_w 100 --lr 3e-4 --batch 8 --det_every 200 --pool_fill $P/fill --pool_cfg $P/cfg --pool_combos lego_jelly --pool_drop 0.0,0.02,0.0,1.5 --pool_frames 80 --phase2 --phys_w 1.0 --save_every 100000"
FLOOR_SP="--bc_level node --bc_mode project --bc_softplus 0.01"
REST=$(echo "$REST_OLD" | sed -e 's/--inv_barrier 1.0 --inv_jhat 0.3 --inv_ext linear/--inv_barrier 0/' -e 's/--det_w 100/--det_w 0 --det_keep_elastic/')
BASE_SP="$COMMON $FLOOR_SP $REST"
BASE_NOXN="$BASE_SP --bc_stick 0.01"
BASE="$BASE_NOXN --bc_stick_xn"
ENVX="AF_SF_RELEX=1 AF_SF_BALANCE=4:4"
SPB=$(P=$P bash -c "$(grep -E '^(COMMON|REST_OLD|FLOOR_SP|REST|BASE_SP|BASE_NOXN|BASE)=' $W/spstick.sh; echo 'echo "$BASE"')")
OUTD=$(comm -23 <(tr ' ' '\n' <<<"$SPB" | sort) <(tr ' ' '\n' <<<"$BASE" | sort) | paste -sd' ')
IND=$(comm -13 <(tr ' ' '\n' <<<"$SPB" | sort) <(tr ' ' '\n' <<<"$BASE" | sort) | paste -sd' ')
echo "[spstick BASE  ] $SPB"
echo "[relexcess BASE] $BASE"
echo "[명령줄 차이 spstick multi -> relexcess] 플래그 빠짐: '$OUTD' | 플래그 더함: '$IND' | 환경변수 더함: $ENVX" | tee -a $U/cmdline_diff.txt
[ "$SPB" = "$BASE" ] || fail "BASE 가 spstick.sh 와 글자 그대로가 아니다"
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
wait_gpu(){ local g=""; while [ -z "$g" ]; do g=$(free_gpu "$1"); [ -z "$g" ] && { echo "[대기] 빈 GPU 없음 $(date +%T)" >> $U/wait.log; sleep 120; }; done; echo $g; }
fit(){ # fit <출력디렉토리> <반복> <저장간격> <gpu> <로그> [평가간격] [추가 env...]
  local O=$1 N=$2 E=$3 G=$4 LG=$5 EV=${6:-100000}; shift 6
  env $ENVX "$@" AF_STATE_FIT_MULTI=$SPEC:$N:8 AF_STATE_FIT_LOSS=ip AF_STATE_FIT_EVERY=$E AF_STATE_FIT_EVAL=$EV AF_STATE_FIT_CK=$O/ck.pt AF_STATE_FIT_TB=$O/tb \
    CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py --infer --data $W/multi5 --hold_traj L000,0000,0001,0002,0003 --out $O/run --tag EV $BASE --iters 0 --eval_t0 0 --eval_len 1 --tb none > $LG 2>&1 < /dev/null; }
# ---- 0) 단위 시험: 새 망(학습 시작점) 필수 통과, 학습 끝 spstick 망(사본)은 L≥0 여부 기록 ----
if [ ! -f $U/unit_test_$C.ok ]; then
  g=$(wait_gpu); echo "[단위 시험] GPU$g $(date +%T)"
  UI=$U/unit_$C/init; UT=$U/unit_$C/trained; [ -e $U/unit_$C ] && mv $U/unit_$C $U/unit_${C}_prev_$(date +%H%M%S); mkdir -p $UI $UT
  S0=$(sha1sum $S/multi/ck.pt | cut -c1-12); cp $S/multi/ck.pt $UT/ck.pt
  fit $UI 15000 500 $g $UI.log 1000 AF_SF_RELEX_TEST=1 AF_SF_RELEX_REF=$GA/states_trained.jsonl
  fit $UT 15000 500 $g $UT.log 1000 AF_SF_RELEX_TEST=1 AF_SF_RELEX_REF=$GA/states_trained.jsonl
  S1=$(sha1sum $S/multi/ck.pt | cut -c1-12); S2=$(sha1sum $UT/ck.pt | cut -c1-12)
  { echo "== 새 망 (학습 시작점) =="; grep -a "공중/접촉\]\|공중 상태 목록\|균형 샘플링\|E\* 캐시\]\|D 분포\|D 하한\|E\* 분포\|시험\|Traceback\|Error" $UI.log
    echo "== 학습 끝 spstick multi 망 (사본, 15000 반복) =="; grep -a "이어 돈다\|시험 \|Traceback\|Error" $UT.log
    echo "spstick multi/ck.pt sha1 전 $S0 후 $S1, 사본 $S2; 새 망 시험 뒤 ck.pt 생김: $([ -e $UI/ck.pt ] && echo 예 || echo 아니오)"; echo "커밋 $C"; } | tee $U/unit_test.txt
  [ "$S0" = "$S1" ] && [ "$S0" = "$S2" ] || fail "spstick ck.pt 가 바뀌었다"
  [ -e $UI/ck.pt ] && fail "시험이 체크포인트를 썼다"
  grep -q RELEX_TEST_OK $UI.log || fail "단위 시험 실패 (새 망)"
  touch $U/unit_test_$C.ok; up
fi
# ---- 1) 이어 돌리기 시험: A = 20 반복(10 마다 저장) 뒤 같은 체크포인트로 30 까지, B = 한 번에 30 -> 비교 ----
if [ ! -f $U/multi_resume_$C.ok ]; then
  g=$(wait_gpu); echo "[이어 돌리기 시험] GPU$g $(date +%T)"
  RT=$U/resume_$C/test; RB=$U/resume_$C/ref; [ -e $U/resume_$C ] && mv $U/resume_$C $U/resume_${C}_prev_$(date +%H%M%S); mkdir -p $RT $RB
  t0=$(date +%s); fit $RT 20 10 $g $RT/r20.log 100000; t1=$(date +%s); fit $RT 30 10 $g $RT/r30.log 100000; t2=$(date +%s)
  fit $RB 30 10 $g $RB/r30.log 100000
  K=$(ck_k $RT/ck.pt); RS=$(grep -a -m1 "이어 돈다" $RT/r30.log | cut -c1-120); CC=$(grep -a -m1 "이전 캐시" $RT/r30.log | cut -c1-160)
  TB=$(cat $RT/r20.log $RT/r30.log $RB/r30.log | grep -a -c Traceback)
  CMP=$(python - $RT/ck.pt $RB/ck.pt <<'PY'
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False); b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
d = max(float((a["net"][k].float() - b["net"][k].float()).abs().max()) for k in a["net"])
s = max(float(b["net"][k].float().abs().max()) for k in b["net"])
o = max(float((x - y).abs().max()) for x, y in zip([t for st in a["opt"]["state"].values() for t in st.values() if torch.is_tensor(t) and t.numel() > 1],
                                                    [t for st in b["opt"]["state"].values() for t in st.values() if torch.is_tensor(t) and t.numel() > 1]))
print(f"반복 {a['k']}/{b['k']}  망 파라미터 최대차 {d:.3e} (최대 크기 {s:.3e})  Adam 상태 최대차 {o:.3e}  생성기 같음 {bool((a['gen'] == b['gen']).all())}")
PY
)
  echo "[이어 돌리기 multi] 30 실행: $RS | 캐시 대조: $CC | 마지막 반복 $K | Traceback $TB | 이어 돈 것 vs 한 번에: $CMP | 20 반복 $((t1-t0)) s, 이어 10 반복 $((t2-t1)) s" | tee $U/resume_test.txt
  [ "$K" = 30 ] && [ -n "$RS" ] && [ "$TB" = 0 ] && grep -q "최대 차 0.000e+00" <<<"$CC" || fail "이어 돌리기 확인 실패 -- 학습 안 함"
  env AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$g python -u exe/train_deform.py --infer --data $W/one/L000 --hold_traj L000 --out $RT/roll --tag EV --resume $RT/ck.pt $BASE --iters 0 --eval_t0 0 --eval_len 3 --tb none > $RT/roll.log 2>&1 < /dev/null
  RL=$(grep -a "롤아웃\] L000" $RT/roll.log | tail -1 | cut -c1-160); echo "[롤아웃 연기 시험] $RL | Traceback $(grep -a -c Traceback $RT/roll.log)" | tee -a $U/resume_test.txt
  [ -n "$RL" ] && [ "$(grep -a -c Traceback $RT/roll.log)" = 0 ] || fail "롤아웃 연기 시험 실패 -- 학습 안 함"
  touch $U/multi_resume_$C.ok; up
fi
# ---- 2) 학습 15000 반복 (500 마다 체크포인트 -> R2 바로, 1000 마다 다섯 궤적 검증 + 공중·접촉 상태 출력 오차) ----
mkdir -p $U/multi
K=$(ck_k $U/multi/ck.pt 2>/dev/null)
if [ "$K" != 15000 ]; then
  g=$(wait_gpu)
  CMD="env $ENVX AF_STATE_FIT_MULTI=$SPEC:15000:8 AF_STATE_FIT_LOSS=ip AF_STATE_FIT_CK=$U/multi/ck.pt AF_STATE_FIT_R2=$R/multi/ck.pt AF_STATE_FIT_TB=$U/multi/tb CUDA_VISIBLE_DEVICES=$g python -u exe/train_deform.py --infer --data $W/multi5 --hold_traj L000,0000,0001,0002,0003 --out $U/multi/run --tag EV $BASE --iters 0 --eval_t0 0 --eval_len 80 --tb none"
  echo "[학습 시작] relexcess multi: 커밋 $C, GPU$g, 이어 돌기 체크포인트 반복 ${K:-없음}, $(date '+%F %T')" | tee -a $U/RELEXCESS_LAUNCHED
  echo "[학습 명령줄] $CMD" | tee -a $U/RELEXCESS_LAUNCHED; up
  $CMD >> $U/multi/run.log 2>&1 < /dev/null
  grep -a "다궤적 롤아웃\|상태 출력 오차\|rror" $U/multi/run.log | tail -8 | cut -c1-200
  K=$(ck_k $U/multi/ck.pt); echo "[multi 학습 끝] 반복 $K $(date +%T)"; up
  [ "$K" = 15000 ] || fail "multi 15000 못 미침 (반복 $K) -- 롤아웃 생략"
fi
# ---- 3) L000·0004 롤아웃 (F float32 덤프) + 3DGS 영상 + psi 색 (GPU 둘 동시에) ----
roll(){ T=$1; g=$2; N=relexcess_multi_$T
  if [ ! -f $U/roll_$N.pt ] || ! grep -a -q "롤아웃\] $T" $U/v_$N.log 2>/dev/null; then rm -f $U/roll_$N.pt $U/inv_$N.jsonl $U/tun_$N.jsonl
    echo "[롤아웃 명령] AF_INV_LOG=$U/inv_$N.jsonl AF_TUNNEL_LOG=$U/tun_$N.jsonl AF_ROLL_DUMP=$U/roll_$N.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$g python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $U/v_$N --tag EV --resume $U/multi/ck.pt $BASE --iters 0 --eval_t0 0 --eval_len 80 --tb none" > $U/cmd_roll_$T.txt
    env AF_INV_LOG=$U/inv_$N.jsonl AF_TUNNEL_LOG=$U/tun_$N.jsonl AF_ROLL_DUMP=$U/roll_$N.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$g \
      python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $U/v_$N --tag EV --resume $U/multi/ck.pt $BASE --iters 0 --eval_t0 0 --eval_len 80 --tb none > $U/v_$N.log 2>&1 < /dev/null
    bgup $U/roll_$N.pt; fi
  grep -a "롤아웃\] $T\|Traceback\|Error" $U/v_$N.log | tail -2 | cut -c1-200
  [ -f $U/roll_$N.pt ] || { echo "[오류] $T 롤아웃 덤프 없음"; return 1; }
  echo "$N $(grep -a '롤아웃\] ' $U/v_$N.log | tail -1 | grep -o '평균 [0-9.]*%')" > $U/err_$N.txt; up
  [ -f $U/h5_$N/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $U/roll_$N.pt --traj $(gt_of $T) --out $U/h5_$N > $U/h5_$N.log 2>&1
  [ -f $U/$N.mp4 ] || CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $U/h5_$N --config $W/lbdata/config.json --shape lego --out $U/$N.mp4 > $U/r_$N.log 2>&1 < /dev/null
  [ -f $U/$N.mp4 ] || { echo "[오류] $T 렌더"; return 1; }
  up
  # psi: psi_color_lb (d993fb5 워크트리, lego_psi_spstick·ipfeatpsi 와 같은 도구·인자), 공통 고정 범위
  D=$U/psi/$T; mkdir -p $D
  ( cd $WP && { [ -f $D/col_rx.npy ] || CUDA_VISIBLE_DEVICES=$g python -u exe/psi_color_lb.py --gt_dir $(gt_of $T) --pt $(pt_of $T) --dump mm=$NS/roll_mm_$T.pt --dump rx=$U/roll_$N.pt --out $D \
        --eps 1e-3 --q 1 99 --cmap turbo --range $RANGE --stats_name defstats_$T.txt --stats_frames 1 40 80 > $D/psi_color.log 2>&1 < /dev/null; }
    grep -q PSI_COLOR_DONE $D/psi_color.log || { tail -5 $D/psi_color.log; exit 1; }
    rm -f $D/col_gt.npy $D/col_mm.npy   # 정답·기존 칸은 lego_psi_spstick 의 같은 범위 렌더를 재사용 (같은 psi·같은 범위로 만든 색)
    [ -f $D/h5_rx/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $U/roll_$N.pt --traj $(gt_of $T) --out $D/h5_rx > $D/h5_rx.log 2>&1
    [ -f $D/psi_rx_lego.mp4 ] || CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $D/h5_rx --config $W/lbdata/config.json --shape lego \
       --colors $D/col_rx.npy --out $D/psi_rx_lego.mp4 > $D/r_rx.log 2>&1 < /dev/null
    [ -f $D/psi_rx_lego.mp4 ] ) || { echo "[오류] $T psi"; return 1; }
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
cat $U/err_relexcess_multi_L000.txt $U/err_relexcess_multi_0004.txt > $U/errors_multi.tmp
# ---- 4) 학습 곡선 (TB -> json): 이번·기존 spstick multi ----
python - $U/multi/tb $S/multi/tb $U/curves.json <<'PY'
import json, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
out = {}
for nm, d in (("relexcess", sys.argv[1]), ("spstick", sys.argv[2])):
    ea = EventAccumulator(d, size_guidance={"scalars": 0}); ea.Reload()
    out[nm] = {k: [(e.step, e.value) for e in ea.Scalars(k)] for k in ea.Tags()["scalars"]
               if k.startswith("다궤적적합/") or k.startswith("다궤적적합_롤아웃_") or k.startswith("다궤적적합_상태/")}
json.dump(out, open(sys.argv[3], "w"), ensure_ascii=False)
print("곡선", {k: len(v) for k, v in out.items()})
PY
grep -a "상태 출력 오차\]" $U/multi/run.log > $U/state_err_log.txt
mv $U/errors_multi.tmp $U/errors_multi.txt; up
kill $SYNC 2>/dev/null
rclone copy $U $R -q && echo "[R2 업로드 완료] $(date)" && touch $U/ALL_DONE $U/RELEXCESS_CLUSTER_DONE && rclone copy $U $R -q
rclone check $U $R --one-way --size-only -q && echo "[R2 대조] 전부 올라감"
echo RELEXCESS_CLUSTER_DONE
