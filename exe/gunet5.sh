#!/bin/bash
# lego 다섯 궤적 상태 적합 -- 몸통만 Graph U-Net (사용자 10-24). 기준 = lbours/noinertial (exe/noinertial.sh, 관성 예측 없음)
#   bash exe/gunet5.sh <GPU>      (잡 안에서만: AF_DETACH=1 bash exe/job_run.sh $W/gunet5_boot.sh <로그>)
#   학습 명령줄 = noinertial 이 실제로 실행한 학습 명령줄(NOINERTIAL_LAUNCHED)에서 경로·GPU 만 바꾸고 XF 하나만 더한 것 (대조, 다르면 멈춤).
#   XF = --gnn_layer unet --gunet_levels 3 (h·2h·4h, 층 2+2+2+1+1 = 8, 폭 128; --gnn_layers 8 은 쓰이지 않는다)
# 순서: 이어 돌리기 시험(20 -> 30 이어 vs 한 번에 30, 망·Adam 비교) -> 롤아웃 연기 시험 -> 15000 반복 (500 마다 체크포인트, R2 즉시)
#       -> L000·0004 롤아웃 (F float32 덤프) + 3DGS 영상 + psi 색 (공통 고정 범위 lbours/lego_psi/range.json) -> 학습 곡선 json
#   GUNET5_ITERS=N (기본 15000): 짧은 시험이면 출력 lbours/gunet5_itN, 학습만 하고 끝 (롤아웃·영상 없음)
# 멱등: 잡이 바뀌어 다시 띄우면 끝난 단계는 건너뛰고 학습은 체크포인트에서 이어 돈다. GPU 는 인자 하나만 쓴다 (다른 실행이 있으면 빌 때까지 기다림).
# 출력은 처음부터 R2 와 통째 동기화 (제외 없음, 5분 루프).
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
G=${1:?GPU}
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
NIT=${GUNET5_ITERS:-15000}
XF="--gnn_layer unet --gunet_levels 3"
TAG=${GUNET5_TAG:-}     # near (사용자 10-24): 바닥 입력 = 접촉 여지 셀 꼭짓점만 [플래그, 거리/h], 법선 없음, 거친 수준에도 -> 출력 gunet5_near
case "$TAG" in "") ;; near) XF="$XF --bc_feat near --bc_near_h 1.0" ;; *) echo "[오류] GUNET5_TAG $TAG"; exit 1 ;; esac
P=$W/lbpool; NI=$W/lbours/noinertial; S=$W/lbours/spstick; NS=$W/lbours/lego_psi_spstick
WP=$W/wt/psi_d993fb5; RANGE=$W/lbours/lego_psi/range.json
DN=gunet5${TAG:+_$TAG}; [ "$NIT" = 15000 ] || DN=${DN}_it$NIT
U=$W/lbours/$DN; R=r2:storage/result/anchorflow/lbours/$DN; mkdir -p $U
if [ -f $U/driver.hb ] && [ $(( $(date +%s) - $(cat $U/driver.hb) )) -lt 300 ]; then echo "[중단] 다른 드라이버가 돌고 있다 (하트비트 5 분 안) $(date)"; exit 0; fi
echo $$ > $U/driver.pid
( while true; do date +%s > $U/driver.hb; sleep 120; done ) & HB=$!
( while true; do rclone copy $U $R -q; sleep 300; done ) & SYNC=$!
trap 'kill $SYNC $HB 2>/dev/null' EXIT
echo "[시작] $(date) 커밋 $C 노드 $(hostname) 잡 ${PBS_JOBID:-?} GPU$G 반복 $NIT 출력 $DN 추가 인자 $XF"
up(){ rclone copy $U $R -q; }
bgup(){ ( rclone copyto $1 $R/${1#$U/} -q && echo "[R2 덤프 올림] ${1#$U/} $(date +%T)" ) & }
fail(){ echo "[오류] $1 $(date)"; echo "$1" > $U/GUNET5_FAIL; up; exit 1; }
rm -f $U/GUNET5_FAIL
# ---- 기준 명령줄: noinertial 의 BASE_NI (스크립트 원문에서 다시 만든 것) 와 실제 실행된 학습 명령줄 둘 다와 대조 ----
BASE_NI=$(P=$P bash -c "$(grep -E '^(COMMON|REST_OLD|FLOOR_SP|REST|BASE_SP|BASE_NOXN|BASE|BASE_NI)=' $W/wt/noinertial_5c2e3a2/exe/noinertial.sh); echo \"\$BASE_NI\"")
[ -n "$BASE_NI" ] || fail "noinertial BASE_NI 못 만듦"
BASE_GU=$(sed "s# --bc_stick_xn# --bc_stick_xn $XF#" <<<"$BASE_NI")
[ "$BASE_GU" = "$BASE_NI $XF" ] || fail "XF 가 끝에 붙지 않았다"
gt_of(){ [ $1 = L000 ] && echo $W/lbdata/test_low/L000 || echo $W/lbdata/test/$1; }
pt_of(){ [ $1 = L000 ] && echo $W/lbpt/test_low/L000.pt || echo $W/lbpt/test/$1.pt; }
cp -n $W/lbdata/test/0004/video.mp4 $U/gt_0004.mp4; cp -n $W/lbdata/test_low/L000/video.mp4 $U/gt_L000.mp4
ck_k(){ python -c "import torch;c=torch.load('$1',map_location='cpu',weights_only=False);print(c.get('k',c.get('it',c.get('step','?'))))" 2>/dev/null; }
SPEC=""; for T in L000 0000 0001 0002 0003; do SPEC="$SPEC${SPEC:+,}$T=$S/roll_ov_$T.pt"; done
for T in L000 0000 0001 0002 0003; do [ -f $S/roll_ov_$T.pt ] || fail "학습 상태 덤프 없음 $S/roll_ov_$T.pt"; done
wait_gpu(){ # 인자 카드에 다른 프로세스가 없을 때까지
  local u; u=$(nvidia-smi -i $G --query-gpu=uuid --format=csv,noheader)
  while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; do echo "[대기] GPU$G 사용 중 $(date +%T)" >> $U/wait.log; sleep 120; done
  echo "[GPU] $G $(date '+%F %T') $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $U/gpu.log; }
fit(){ # fit <출력디렉토리> <반복> <저장간격> <로그>
  local O=$1 N=$2 E=$3 LG=$4
  env AF_STATE_FIT_MULTI=$SPEC:$N:8 AF_STATE_FIT_LOSS=ip AF_STATE_FIT_EVERY=$E AF_STATE_FIT_EVAL=100000 AF_STATE_FIT_CK=$O/ck.pt AF_STATE_FIT_TB=$O/tb \
    CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py --infer --data $W/multi5 --hold_traj L000,0000,0001,0002,0003 --out $O/run --tag EV $BASE_GU --iters 0 --eval_t0 0 --eval_len 1 --tb none > $LG 2>&1 < /dev/null; }
# ---- 1) 이어 돌리기 시험: A = 20 반복(10 마다 저장) 뒤 같은 체크포인트로 30 까지, B = 한 번에 30 -> 비교 ----
if [ ! -f $U/multi_resume_$C.ok ]; then
  wait_gpu; echo "[이어 돌리기 시험] GPU$G $(date +%T)"
  RT=$U/resume_$C/test; RB=$U/resume_$C/ref; [ -e $U/resume_$C ] && mv $U/resume_$C $U/resume_${C}_prev_$(date +%H%M%S); mkdir -p $RT $RB
  t0=$(date +%s); fit $RT 20 10 $RT/r20.log; t1=$(date +%s); fit $RT 30 10 $RT/r30.log; t2=$(date +%s)
  fit $RB 30 10 $RB/r30.log
  K=$(ck_k $RT/ck.pt); RS=$(grep -a -m1 "이어 돈다" $RT/r30.log | cut -c1-120); NF=$(grep -a -m1 "Graph U-Net\] 파라미터" $RT/r20.log | cut -c1-200)
  TB=$(cat $RT/r20.log $RT/r30.log $RB/r30.log | grep -a -c Traceback)
  CMP=$(python - $RT/ck.pt $RB/ck.pt <<'PY'
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False); b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
d = max(float((a["net"][k].float() - b["net"][k].float()).abs().max()) for k in a["net"])
s = max(float(b["net"][k].float().abs().max()) for k in b["net"])
o = max(float((x - y).abs().max()) for x, y in zip([t for st in a["opt"]["state"].values() for t in st.values() if torch.is_tensor(t) and t.numel() > 1],
                                                    [t for st in b["opt"]["state"].values() for t in st.values() if torch.is_tensor(t) and t.numel() > 1]))
print(f"반복 {a['k']}/{b['k']}  입력폭 {a['n_feat']}  스킵 결합 키 {sum('fuse' in k for k in a['net'])}  망 파라미터 최대차 {d:.3e} (최대 크기 {s:.3e})  "
      f"Adam 상태 최대차 {o:.3e}  생성기 같음 {bool((a['gen'] == b['gen']).all())}")
PY
)
  echo "[이어 돌리기 gunet5] 30 실행: $RS | 마지막 반복 $K | Traceback $TB | 이어 돈 것 vs 한 번에: $CMP | 20 반복 $((t1-t0)) s, 이어 10 반복 $((t2-t1)) s | $NF" | tee $U/resume_test.txt
  NFW=33; [ "$TAG" = near ] && NFW=31     # 입력폭: 바닥 dist 4 칸 -> near 2 칸 (플래그, 거리/h)
  [ "$K" = 30 ] && [ -n "$RS" ] && [ "$TB" = 0 ] && grep -q "입력폭 $NFW " <<<"$CMP" && grep -q "생성기 같음 True" <<<"$CMP" && ! grep -q "스킵 결합 키 0 " <<<"$CMP" || fail "이어 돌리기 확인 실패 -- 학습 안 함"
  python -c "import re,sys;m=re.search(r'망 파라미터 최대차 ([0-9.e+-]+)', sys.argv[1]);sys.exit(0 if m and float(m.group(1))<1e-3 else 1)" "$CMP" || fail "이어 돈 것과 한 번에 돈 것의 망 차이가 1e-3 이상"
  env AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py --infer --data $W/one/L000 --hold_traj L000 --out $RT/roll --tag EV --resume $RT/ck.pt $BASE_GU --iters 0 --eval_t0 0 --eval_len 3 --tb none > $RT/roll.log 2>&1 < /dev/null
  RL=$(grep -a "롤아웃\] L000" $RT/roll.log | tail -1 | cut -c1-160); echo "[롤아웃 연기 시험] $RL | Traceback $(grep -a -c Traceback $RT/roll.log)" | tee -a $U/resume_test.txt
  [ -n "$RL" ] && [ "$(grep -a -c Traceback $RT/roll.log)" = 0 ] || fail "롤아웃 연기 시험 실패 -- 학습 안 함"
  touch $U/multi_resume_$C.ok; up
fi
# ---- 2) 학습 NIT 반복 (500 마다 체크포인트 -> R2 바로, 1000 마다 다섯 궤적 검증 -- noinertial 과 같은 환경 변수) ----
mkdir -p $U/multi
K=$(ck_k $U/multi/ck.pt 2>/dev/null)
if [ "$K" != $NIT ]; then
  wait_gpu
  CMD="env AF_STATE_FIT_MULTI=$SPEC:$NIT:8 AF_STATE_FIT_LOSS=ip AF_STATE_FIT_CK=$U/multi/ck.pt AF_STATE_FIT_R2=$R/multi/ck.pt AF_STATE_FIT_TB=$U/multi/tb CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py --infer --data $W/multi5 --hold_traj L000,0000,0001,0002,0003 --out $U/multi/run --tag EV $BASE_GU --iters 0 --eval_t0 0 --eval_len 80 --tb none"
  # 대조: noinertial 이 실제 실행한 학습 명령줄에서 경로·GPU·반복 수만 바꾸고 XF 만 더한 것과 글자 그대로 같아야 한다
  NIC=$(grep -a -m1 "학습 명령줄" $NI/NOINERTIAL_LAUNCHED | sed -e 's/^\[학습 명령줄\] //' -e "s#lbours/noinertial/#lbours/$DN/#g" -e "s#:15000:8 #:$NIT:8 #" \
        -e "s# --bc_stick_xn # --bc_stick_xn $XF #" -e 's/CUDA_VISIBLE_DEVICES=[0-9]*/CUDA_VISIBLE_DEVICES=X/')
  [ -n "$NIC" ] && [ "$NIC" = "$(sed 's/CUDA_VISIBLE_DEVICES=[0-9]*/CUDA_VISIBLE_DEVICES=X/' <<<"$CMD")" ] \
    && echo "[대조] 학습 명령줄 == noinertial 실행 학습 명령줄 + '$XF' (경로·GPU·반복 수만 다름)" | tee -a $U/cmdline_diff.txt || fail "학습 명령줄이 noinertial 명령줄 + XF 와 다르다"
  echo "[학습 시작] gunet5: 커밋 $C, $XF, GPU$G, 이어 돌기 체크포인트 반복 ${K:-없음}, $(date '+%F %T')" | tee -a $U/GUNET5_LAUNCHED
  echo "[학습 명령줄] $CMD" | tee -a $U/GUNET5_LAUNCHED; up
  $CMD >> $U/multi/run.log 2>&1 < /dev/null
  grep -a "다궤적 롤아웃\|rror" $U/multi/run.log | tail -8 | cut -c1-200
  K=$(ck_k $U/multi/ck.pt); echo "[multi 학습 끝] 반복 $K $(date +%T)"; up
  [ "$K" = $NIT ] || fail "multi $NIT 못 미침 (반복 $K) -- 롤아웃 생략"
fi
if [ "$NIT" != 15000 ]; then touch $U/GUNET5_SMOKE_DONE; up; echo "GUNET5_SMOKE_DONE $NIT"; exit 0; fi
# ---- 3) L000·0004 롤아웃 (F float32 덤프) + 3DGS 영상 + psi 색 (같은 GPU 에서 차례로; noinertial 과 같은 명령) ----
roll(){ T=$1; g=$G; N=gunet5_multi_$T
  if [ ! -f $U/roll_$N.pt ] || ! grep -a -q "롤아웃\] $T" $U/v_$N.log 2>/dev/null; then rm -f $U/roll_$N.pt $U/inv_$N.jsonl $U/tun_$N.jsonl
    echo "[롤아웃 명령] AF_INV_LOG=$U/inv_$N.jsonl AF_TUNNEL_LOG=$U/tun_$N.jsonl AF_ROLL_DUMP=$U/roll_$N.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$g python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $U/v_$N --tag EV --resume $U/multi/ck.pt $BASE_GU --iters 0 --eval_t0 0 --eval_len 80 --tb none" > $U/cmd_roll_$T.txt
    env AF_INV_LOG=$U/inv_$N.jsonl AF_TUNNEL_LOG=$U/tun_$N.jsonl AF_ROLL_DUMP=$U/roll_$N.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$g \
      python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $U/v_$N --tag EV --resume $U/multi/ck.pt $BASE_GU --iters 0 --eval_t0 0 --eval_len 80 --tb none > $U/v_$N.log 2>&1 < /dev/null
    bgup $U/roll_$N.pt; fi
  grep -a "롤아웃\] $T\|Traceback\|Error" $U/v_$N.log | tail -2 | cut -c1-200
  [ -f $U/roll_$N.pt ] || { echo "[오류] $T 롤아웃 덤프 없음"; return 1; }
  echo "$N $(grep -a '롤아웃\] ' $U/v_$N.log | tail -1 | grep -o '평균 [0-9.]*%')" > $U/err_$N.txt; up
  [ -f $U/h5_$N/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $U/roll_$N.pt --traj $(gt_of $T) --out $U/h5_$N > $U/h5_$N.log 2>&1
  [ -f $U/$N.mp4 ] || CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $U/h5_$N --config $W/lbdata/config.json --shape lego --out $U/$N.mp4 > $U/r_$N.log 2>&1 < /dev/null
  [ -f $U/$N.mp4 ] || { echo "[오류] $T 렌더"; return 1; }
  up
  # psi: psi_color_lb (d993fb5 워크트리, noinertial 과 같은 도구·인자), 공통 고정 범위; 칸 ni = 기준 noinertial, gu = 이번
  D=$U/psi/$T; mkdir -p $D
  ( cd $WP && { [ -f $D/col_gu.npy ] || CUDA_VISIBLE_DEVICES=$g python -u exe/psi_color_lb.py --gt_dir $(gt_of $T) --pt $(pt_of $T) --dump ni=$NI/roll_noinertial_multi_$T.pt --dump gu=$U/roll_$N.pt --out $D \
        --eps 1e-3 --q 1 99 --cmap turbo --range $RANGE --stats_name defstats_$T.txt --stats_frames 1 40 80 > $D/psi_color.log 2>&1 < /dev/null; }
    grep -q PSI_COLOR_DONE $D/psi_color.log || { tail -5 $D/psi_color.log; exit 1; }
    [ -f $D/h5_gu/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $U/roll_$N.pt --traj $(gt_of $T) --out $D/h5_gu > $D/h5_gu.log 2>&1
    [ -f $D/psi_gu_lego.mp4 ] || CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $D/h5_gu --config $W/lbdata/config.json --shape lego \
       --colors $D/col_gu.npy --out $D/psi_gu_lego.mp4 > $D/r_gu.log 2>&1 < /dev/null
    [ -f $D/psi_gu_lego.mp4 ] ) || { echo "[오류] $T psi"; return 1; }
  grep -a "\[색 범위\]" $D/psi_color.log; cat $D/defstats_$T.txt
  # 정답·기준 칸은 같은 psi·같은 범위로 이미 만든 렌더를 재사용 (noinertial psi/<T>)
  cp $NI/psi/$T/psi_gt_lego.mp4 $D/psi_gt_lego.mp4; cp $NI/psi/$T/psi_ni_lego.mp4 $D/psi_ni_lego.mp4; cp $NI/psi/$T/range.json $D/range_noinertial.json
  touch $U/.done_roll_$T; }
for T in L000 0004; do
  [ -f $U/.done_roll_$T ] || { wait_gpu; roll $T > $U/task_roll_$T.log 2>&1; cat $U/task_roll_$T.log | cut -c1-200; }
  [ -f $U/.done_roll_$T ] || fail "롤아웃·영상·psi 단계 실패 ($T)"
done
cat $U/err_gunet5_multi_L000.txt $U/err_gunet5_multi_0004.txt > $U/errors_multi.txt
# ---- 4) 학습 곡선 (TB -> json): 이번·기준 noinertial ----
python - $U/multi/tb $NI/multi/tb $U/curves.json <<'PY'
import json, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
out = {}
for nm, d in (("gunet5", sys.argv[1]), ("noinertial", sys.argv[2])):
    ea = EventAccumulator(d, size_guidance={"scalars": 0}); ea.Reload()
    out[nm] = {k: [(e.step, e.value) for e in ea.Scalars(k)] for k in ea.Tags()["scalars"]
               if k.startswith("다궤적적합/") or k.startswith("다궤적적합_롤아웃_")}
json.dump(out, open(sys.argv[3], "w"), ensure_ascii=False)
print("곡선", {k: len(v) for k, v in out.items()})
PY
up
kill $SYNC 2>/dev/null
rclone copy $U $R -q && echo "[R2 업로드 완료] $(date)" && touch $U/ALL_DONE $U/GUNET5_CLUSTER_DONE && rclone copy $U $R -q
rclone check $U $R --one-way --size-only -q && echo "[R2 대조] 전부 올라감"
echo GUNET5_CLUSTER_DONE
