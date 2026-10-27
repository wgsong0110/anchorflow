#!/bin/bash
# 셀 응력 모델 (사용자 10-27 확정 설계, doc/stress_lego.md, exe/train_stress.py, lib/anchorflow/stress_net.py).
#   bash exe/stress_lego.sh test <GPU>              실제 상태 캐시 + 분포 비교 (cover) -> 보존·회전 공변·mom_core 규약 검산 (check)
#                                                   -> 이어 돌리기 시험 (20 -> 30 vs 한 번에 30) -> 짧은 학습 (bary K1, bspline2 K4 각 ST_SMOKE_IT, 기본 300) + 평가 -> 추론 시간
#                                                   출력 lbours/stress_lego/test_<커밋>, 표식 STRESS_TEST_DONE / STRESS_TEST_FAIL
#   bash exe/stress_lego.sh run <interp> <K> <GPU>  본 학습 (ST_ITERS, 기본 100000) -> 최저 체크포인트 평가 (eval.json) -> 추론 시간 (bench.json)
#                                                   출력 lbours/stress_lego/run_<interp>_K<K>, 표식 STRESS_<interp>_K<K>_LAUNCHED / _DONE / _FAIL
#   잡 안에서만: AF_DETACH=1 bash exe/job_run.sh <부트 스크립트> <로그>. 멱등: 다시 띄우면 끝난 단계는 건너뛰고 학습은 ck.pt 에서 이어 돈다.
#   출력은 처음부터 R2 와 통째 동기화 (제외 없음, 3 분 루프; 체크포인트·결과 json 은 저장 즉시 따로 복사).
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
STAGE=${1:?test 또는 run}
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
U=$W/lbours/stress_lego; R=r2:storage/result/anchorflow/lbours/stress_lego; mkdir -p $U
REAL=$U/real_states.pt
case $STAGE in
  test) G=${2:?GPU}; O=$U/test_$C; MK=STRESS_TEST ;;
  run) IP=${2:?interp}; KQ=${3:?K}; G=${4:?GPU}; O=$U/run_${IP}_K${KQ}; MK=STRESS_${IP}_K${KQ} ;;
  *) echo "[오류] 단계 $STAGE"; exit 1 ;;
esac
RO=$R/${O#$U/}; mkdir -p $O
up(){ rclone copy $O $RO -q; }
fail(){ echo "[오류] $1 $(date)"; echo "$(date '+%m-%d %H:%M') $1" >> $O/${MK}_FAIL; up; exit 1; }
if [ -f $O/driver.hb ] && [ $(( $(date +%s) - $(cat $O/driver.hb) )) -lt 300 ]; then echo "[중단] 다른 드라이버가 돌고 있다 $(date)"; exit 0; fi
echo $$ > $O/driver.pid
( while true; do date +%s > $O/driver.hb; sleep 120; done ) & HB=$!
( while true; do sleep 180; rclone copy $O $RO -q; done ) & SYNC=$!
trap 'kill $SYNC $HB 2>/dev/null; rm -f $O/driver.hb' EXIT
rm -f $O/${MK}_FAIL
echo "[시작] $(date) 단계 $STAGE 커밋 $C 노드 $(hostname) 잡 ${PBS_JOBID:-?} GPU$G | $(df -h /home | tail -1)"
while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 500 ]; do echo "[대기] GPU$G 사용 중 $(date +%T)"; sleep 120; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $O/gpu_at_start.txt
TS="python -u exe/train_stress.py --commit $C --real $REAL"
ck_it(){ python -c "import torch;print(torch.load('$1',map_location='cpu',weights_only=False)['it'])" 2>/dev/null; }
GENJ=${ST_GEN:-'{}'}

if [ $STAGE = test ]; then
  # 1) 실제 상태 캐시 + 분포 비교
  if ! grep -q STRESS_STATS_DONE $O/stats/stats.log 2>/dev/null; then mkdir -p $O/stats
    CUDA_VISIBLE_DEVICES=$G $TS --mode stats --out $O/stats --r2 $RO/stats --gen "$GENJ" > $O/stats/stats.log 2>&1 < /dev/null
    [ -f $REAL ] && rclone copyto $REAL $R/real_states.pt -q; fi
  grep -a "\[실제\]\|롤아웃 덤프\|dump40:" $O/stats/stats.log | cut -c1-300; cat $O/stats/cover.txt
  grep -q STRESS_STATS_DONE $O/stats/stats.log || fail "stats 실패 $(grep -a 'Traceback\|Error' $O/stats/stats.log | tail -2)"
  up
  # 2) 검산
  if ! grep -q STRESS_CHECK_DONE $O/check/check.log 2>/dev/null; then mkdir -p $O/check
    CUDA_VISIBLE_DEVICES=$G $TS --mode check --interp bary --K 1 --out $O/check --r2 $RO/check --gen "$GENJ" > $O/check/check.log 2>&1 < /dev/null; fi
  cat $O/check/check.txt
  grep -q STRESS_CHECK_DONE $O/check/check.log || fail "check 실패 $(grep -a 'Traceback\|Error' $O/check/check.log | tail -2)"
  up
  # 3) 이어 돌리기: A = 20 (10 마다 저장) 뒤 30 까지 이어서, B = 한 번에 30 -> 망·Adam·생성기 비교
  if [ ! -f $O/resume.ok ]; then
    RA=$O/resA; RB=$O/resB; rm -rf $RA $RB; mkdir -p $RA $RB
    RS="--mode train --interp bspline2 --K 4 --batch 256 --ckpt_every 10 --eval_every 100000 --sched_iters 30 --warm 5 --log_every 5 --no_real_eval --n_test 1024"
    CUDA_VISIBLE_DEVICES=$G $TS $RS --gen "$GENJ" --iters 20 --out $RA > $RA/a20.log 2>&1 < /dev/null
    CUDA_VISIBLE_DEVICES=$G $TS $RS --gen "$GENJ" --iters 30 --out $RA > $RA/a30.log 2>&1 < /dev/null
    CUDA_VISIBLE_DEVICES=$G $TS $RS --gen "$GENJ" --iters 30 --out $RB > $RB/b30.log 2>&1 < /dev/null
    CMP=$(python - $RA/ck.pt $RB/ck.pt <<'PY'
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False); b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
d = max(float((a["net"][k].float() - b["net"][k].float()).abs().max()) for k in a["net"])
s = max(float(b["net"][k].float().abs().max()) for k in b["net"])
o = max(float((x - y).abs().max()) for x, y in zip([t for st in a["opt"]["state"].values() for t in st.values() if torch.is_tensor(t) and t.numel() > 1],
                                                    [t for st in b["opt"]["state"].values() for t in st.values() if torch.is_tensor(t) and t.numel() > 1]))
print(f"반복 {a['it']}/{b['it']} 망 파라미터 최대차 {d:.3e} (최대 크기 {s:.3e}) Adam 상태 최대차 {o:.3e} 생성기 같음 {bool((a['gen'] == b['gen']).all())} "
      f"CPU 난수 같음 {bool((a['cpu_rng'] == b['cpu_rng']).all())} 평가 기록 {len(a['hist'])}/{len(b['hist'])}")
PY
)
    LA=$(grep -a "\[평가\]\|무작위 시험" $RA/a30.log | tail -1 | cut -c1-200); LB=$(grep -a "무작위 시험" $RB/b30.log | tail -1 | cut -c1-200)
    RSL=$(grep -a -m1 "이어 돈다" $RA/a30.log | cut -c1-120); TB_=$(cat $RA/*.log $RB/*.log | grep -a -c Traceback)
    echo "[이어 돌리기] $RSL | $CMP | Traceback $TB_" | tee $O/resume_test.txt
    echo "  A 마지막 평가: $LA" | tee -a $O/resume_test.txt; echo "  B 마지막 평가: $LB" | tee -a $O/resume_test.txt
    grep -q "반복 30/30" <<<"$CMP" && grep -q "생성기 같음 True" <<<"$CMP" && [ -n "$RSL" ] && [ "$TB_" = 0 ] || fail "이어 돌리기 확인 실패"
    python -c "import re,sys;m=re.search(r'망 파라미터 최대차 ([0-9.e+-]+)', sys.argv[1]);sys.exit(0 if m and float(m.group(1))<1e-4 else 1)" "$CMP" || fail "이어 돈 것과 한 번에 돈 것의 망 차이 1e-4 이상"
    touch $O/resume.ok; up
  fi
  # 4) 짧은 학습 (두 보간) + 평가 + 추론 시간
  NS=${ST_SMOKE_IT:-300}
  for IK in "bary 1" "bspline2 4"; do
    set -- $IK; SM=$O/smoke_$1_K$2; mkdir -p $SM
    if [ "$(ck_it $SM/ck.pt)" != "$NS" ]; then
      echo "[짧은 학습 명령] CUDA_VISIBLE_DEVICES=$G $TS --mode train --interp $1 --K $2 --iters $NS --eval_every $NS --ckpt_every 100 --log_every 10 --gen '$GENJ' --out $SM --r2 $RO/${SM#$O/}" > $SM/cmd.txt
      CUDA_VISIBLE_DEVICES=$G $TS --mode train --interp $1 --K $2 --iters $NS --eval_every $NS --ckpt_every 100 --log_every 10 --gen "$GENJ" --out $SM --r2 $RO/${SM#$O/} >> $SM/train.log 2>&1 < /dev/null
    fi
    grep -a "\[설정\]\|\[반복\|\[평가\]\|무작위 시험\|lego 실제\|dump40 \|Traceback\|Error" $SM/train.log | cut -c1-330 | tail -8
    [ "$(ck_it $SM/ck.pt)" = "$NS" ] || fail "짧은 학습 실패 $1 K$2"
    if ! grep -q STRESS_BENCH_DONE $SM/bench.log 2>/dev/null; then
      CUDA_VISIBLE_DEVICES=$G $TS --mode bench --interp $1 --K $2 --resume $SM/ck.pt --out $SM --r2 $RO/${SM#$O/} > $SM/bench.log 2>&1 < /dev/null; fi
    grep -a "\[추론 시간\]" $SM/bench.log | cut -c1-200
    grep -q STRESS_BENCH_DONE $SM/bench.log || fail "추론 시간 실패 $1 K$2"
  done
  touch $O/${MK}_DONE; up; echo ${MK}_DONE; exit 0
fi

# ---------------- run: 본 학습 ----------------
NIT=${ST_ITERS:-100000}
[ -f $REAL ] || fail "실제 상태 캐시 없음 $REAL (test 단계 먼저)"
K=$(ck_it $O/ck.pt)
if [ "$K" != "$NIT" ]; then
  CMD="CUDA_VISIBLE_DEVICES=$G $TS --mode train --interp $IP --K $KQ --iters $NIT --batch ${ST_BATCH:-1024} --eval_every ${ST_EVAL:-2000} --ckpt_every 1000 --gen '$GENJ' --out $O --r2 $RO"
  echo "[학습 시작] 커밋 $C GPU$G 이어 돌 반복 ${K:-없음} $(date '+%F %T')" | tee -a $O/${MK}_LAUNCHED
  echo "[학습 명령줄] $CMD" | tee -a $O/${MK}_LAUNCHED; up
  eval "$CMD" >> $O/train.log 2>&1 < /dev/null
  K=$(ck_it $O/ck.pt); echo "[학습 끝] 반복 $K $(date +%T)"; up
  [ "$K" = "$NIT" ] || fail "학습이 $NIT 에 못 미침 (반복 $K) $(grep -a 'Traceback\|Error' $O/train.log | tail -2)"
fi
if ! grep -q STRESS_EVAL_DONE $O/eval.log 2>/dev/null; then
  CUDA_VISIBLE_DEVICES=$G $TS --mode eval --interp $IP --K $KQ --resume $O/ck_best.pt --gen "$GENJ" --out $O --r2 $RO > $O/eval.log 2>&1 < /dev/null; fi
grep -q STRESS_EVAL_DONE $O/eval.log || fail "평가 실패 $(grep -a 'Traceback\|Error' $O/eval.log | tail -2)"
if ! grep -q STRESS_BENCH_DONE $O/bench.log 2>/dev/null; then
  CUDA_VISIBLE_DEVICES=$G $TS --mode bench --interp $IP --K $KQ --resume $O/ck_best.pt --out $O --r2 $RO > $O/bench.log 2>&1 < /dev/null; fi
grep -q STRESS_BENCH_DONE $O/bench.log || fail "추론 시간 실패"
kill $SYNC 2>/dev/null
{ echo "${MK}_DONE $(date) 커밋 $C"; grep -a "\[평가\]" -A3 $O/eval.log | tail -4; grep -a "\[추론 시간\]" $O/bench.log; } > $O/${MK}_DONE; up
rclone check $O $RO --one-way --size-only -q && echo "[R2 대조] 전부 올라감"
echo "${MK}_DONE $(date)"
