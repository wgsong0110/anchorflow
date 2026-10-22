#!/bin/bash
# 프레임별 맞춤 (사용자 10-22, 재실행): "모델·증분 포텐셜 목적함수로 한 프레임은 맞출 수 있는데 여러 프레임을 함께 학습하면 간섭 때문에 안 된다" 확인.
#   궤적 T 마다 프레임 t = 0..79:
#     1) 망 갱신: 출력만 최적화 상태 t (lbours/spstick/roll_ov_T.pt) 하나만으로 학습과 같은 한 스텝 손실(phys_w·E + det 벌점)로 고정 반복 Adam
#     2) 옮기기: 그 망으로 모델 자신의 롤아웃 상태에서 한 스텝 (출력만 최적화 상태 안 씀)
#   망: 프레임 0 = 새 망(출력층 0 초기화, 입력 표준화 = noinertial 다섯 궤적 학습 체크포인트의 것), t>0 = 직전 프레임 망·Adam 에서 이어서.
#   새 규칙 (사용자 10-22): **--net_inertial 뺌** (노드 변위 = 망 출력 그대로). BASE_NI = spstick BASE 에서 ' --net_inertial' 만 지운 문자열 (대조 기록).
#   궤적: 0004 먼저, L000 은 그 뒤 (같은 GPU 목록에서). GPU: PF_GPUS (기본 "0") 안에서 빈 카드만 (1~6 은 pf3obj).
#   구현: exe/train_deform.py _perframe_fit (AF_PERFRAME_FIT). 이미 통과한 시험(기본 경로 대조 2ff333a/bf8dc97, --net_inertial 판 재현) 은 다시 안 함;
#   --net_inertial 뺀 경로로 (1) 재현 시험 (반복 0 + noinertial 망 -> roll_noinertial_multi_0004 와 같은 상태, 3 프레임)
#                          (2) 짧은 맞춤 + 이어 돌리기 시험 (3 프레임 x 20 반복: 한 번에 vs 2 프레임 뒤 이어서)
#   E* = 출력만 최적화 상태별 최솟값 -- 그 상태들을 만든 출력만 최적화 실행(spstick.sh ov: BASE $OV, --net_inertial 포함) 설정 그대로 (AF_OV_MIN_STATES)
# 출력 $W/lbours/perframe_fit/ni (R2 lbours/perframe_fit/ni) -- 처음부터 통째 동기화(5 분, 제외 없음), 프레임 파일은 생기는 즉시 올림.
# 동기화·하트비트 루프는 PID 로만 관리하고 wait 로 기다리지 않는다. 잡 안에서만: AF_DETACH=1 bash exe/job_run.sh $W/perframe_boot.sh <로그>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
PF_GPUS=${PF_GPUS:-0}
P=$W/lbpool; U=$W/lbours/perframe_fit/ni; S=$W/lbours/spstick; NS=$W/lbours/lego_psi_spstick; NI=$W/lbours/noinertial
WP=$W/wt/psi_d993fb5; RANGE=$W/lbours/lego_psi/range.json; MCK=$NI/multi/ck.pt
R=r2:storage/result/anchorflow/lbours/perframe_fit/ni; mkdir -p $U
if [ -f $U/driver.hb ] && [ $(( $(date +%s) - $(cat $U/driver.hb) )) -lt 300 ]; then echo "[중단] 다른 드라이버가 돌고 있다 (하트비트 5 분 안) $(date)"; exit 0; fi
echo $$ > $U/driver.pid
( while true; do date +%s > $U/driver.hb; sleep 120; done ) & HB=$!
( while true; do rclone copy $U $R -q; sleep 300; done ) & SYNC=$!
trap 'kill $SYNC $HB 2>/dev/null' EXIT
echo "[시작] $(date) 커밋 $C 노드 $(hostname) 잡 ${PBS_JOBID:-?} GPU 목록 $PF_GPUS"
up(){ rclone copy $U $R -q --include "*.mp4" --include "*.txt" --include "*.json" --include "*.jsonl" --include "*.log" --include "PERFRAME_*" --max-depth 2; }
fail(){ echo "[오류] $1 $(date)"; echo "$1" > $U/PERFRAME_FAIL; up; exit 1; }
BASE=$(P=$P bash -c "$(grep -E '^(COMMON|REST_OLD|FLOOR_SP|REST|BASE_SP|BASE_NOXN|BASE)=' $W/spstick.sh; echo 'echo "$BASE"')")
OV=$(bash -c "$(grep -E '^OV=' $W/spstick.sh); echo \"\$OV\"")
[ -n "$BASE" ] && [ "$OV" = "--out_var --ov_init predict --ov_roll 500 --ov_opt lbfgs --ov_vars out --ov_obj ip" ] || fail "spstick.sh 에서 BASE/OV 를 못 만들었다"
[ "[BASE] $BASE" = "$(grep -a -m1 '^\[BASE\]' $W/spstick.out)" ] || fail "BASE 가 spstick.out 과 다르다"
BASE_NI="${BASE/ --net_inertial / }"
OUTD=$(comm -23 <(tr ' ' '\n' <<<"$BASE" | sort) <(tr ' ' '\n' <<<"$BASE_NI" | sort) | paste -sd' ')
IND=$(comm -13 <(tr ' ' '\n' <<<"$BASE" | sort) <(tr ' ' '\n' <<<"$BASE_NI" | sort) | paste -sd' ')
[ "$OUTD" = "--net_inertial" ] && [ -z "$IND" ] && [ $(echo $BASE | wc -w) = $(( $(echo $BASE_NI | wc -w) + 1 )) ] || fail "BASE_NI 가 BASE 에서 --net_inertial 하나만 뺀 것이 아니다"
NIB=$(grep -a -m1 "학습 명령줄" $NI/NOINERTIAL_LAUNCHED | grep -o -- "--tag EV .* --iters 0" | sed 's/^--tag EV //; s/ --iters 0$//')
[ "$NIB" = "$BASE_NI" ] || fail "BASE_NI 가 noinertial 학습 명령줄의 BASE 와 다르다"
gt_of(){ [ $1 = L000 ] && echo $W/lbdata/test_low/L000 || echo $W/lbdata/test/$1; }
pt_of(){ [ $1 = L000 ] && echo $W/lbpt/test_low/L000.pt || echo $W/lbpt/test/$1.pt; }
for T in 0004 L000; do [ -f $S/roll_ov_$T.pt ] || fail "출력만 최적화 덤프 없음 $T"; [ -f $NI/roll_noinertial_multi_$T.pt ] || fail "noinertial 다섯 궤적 덤프 없음 $T"; done
[ -f $MCK ] || fail "noinertial 다섯 궤적 체크포인트 없음 $MCK"
pfcmd(){ # pfcmd <T> <반복> <출력디렉토리> <gpu> [추가 env]
  local T=$1 N=$2 O=$3 g=$4; shift 4
  echo "env $* AF_PERFRAME_FIT=$S/roll_ov_$T.pt:$N AF_PERFRAME_OUT=$O AF_PERFRAME_STATS=$MCK CUDA_VISIBLE_DEVICES=$g python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $O/run --tag PF $BASE_NI --iters 0 --eval_t0 0 --eval_len 80 --tb none"; }
{ echo "[spstick BASE (기존 프레임별 맞춤·다섯 궤적 학습)] $BASE"
  echo "[이번 BASE_NI] $BASE_NI"
  echo "[명령줄 차이 BASE -> BASE_NI] 플래그 빠짐: '$OUTD' | 플래그 더함: '$IND' (낱말 수 $(echo $BASE | wc -w) -> $(echo $BASE_NI | wc -w))"
  echo "[대조] BASE_NI == noinertial 다섯 궤적 학습 명령줄의 BASE (글자 그대로)"
  echo "[프레임별 맞춤 명령줄 0004 1000] $(pfcmd 0004 1000 $U/n1000/0004 X AF_PERFRAME_R2=$R/n1000/0004)"
  echo "[E* 명령줄 설정] 출력만 최적화 상태를 만든 spstick.sh ov 와 같은 BASE(--net_inertial 포함) $OV"; } | tee $U/cmdline_diff.txt
free_gpu(){ # PF_GPUS 안의 빈 GPU (메모리 1GB 미만 + 프로세스 없음), 인자 카드는 뺀다
  for c in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<1024{print $1}'); do
    case " $PF_GPUS " in *" $c "*) ;; *) continue ;; esac
    case " $* " in *" $c "*) continue ;; esac
    u=$(nvidia-smi -i $c --query-gpu=uuid --format=csv,noheader)
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u" && continue
    echo $c; return; done; }
wait_gpu(){ local g=""; while [ -z "$g" ]; do g=$(free_gpu "$@"); [ -z "$g" ] && { echo "[대기] 빈 GPU 없음 ($PF_GPUS) $(date +%T)" >> $U/wait.log; sleep 120; }; done
  echo "[GPU 배정] $g $(date '+%F %T') $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $U/gpu.log; echo $g; }
# ---- 1) 재현 시험 (--net_inertial 뺌): 반복 0 + 프레임 0 망 = noinertial 다섯 궤적 망 -> roll_noinertial_multi_0004 와 같은 상태 ----
#   F 는 J·F 누적이라 사면체 경계 입자는 위치 1e-8 차이로 다른 J 를 받을 수 있다 -> 입자별 F 차 평균(<1e-5)·1e-3 넘는 비율(<0.1%)
if [ ! -f $U/repro_$C.ok ]; then
  g=$(wait_gpu); RP=$U/test_$C/repro; [ -e $RP ] && mv $RP ${RP}_prev_$(date +%H%M%S); mkdir -p $RP
  CMD=$(pfcmd 0004 0 $RP $g AF_PERFRAME_INIT=$MCK | sed 's/--eval_len 80/--eval_len 3/'); echo "[재현 시험 명령] $CMD" | tee $RP/cmd.txt
  $CMD > $RP/run.log 2>&1 < /dev/null
  RES=$(python - $RP/frames $NI/roll_noinertial_multi_0004.pt <<'PY'
import sys, os, torch
D = torch.load(sys.argv[2], map_location="cpu", weights_only=False); ext = float(D["EXT"])
out = []
for t in range(3):
    c = torch.load(os.path.join(sys.argv[1], f"frame_{t:02d}.pt"), map_location="cpu", weights_only=False)
    dx = float((c["state_out"]["x"] - D["pred"][t].float()).norm(dim=-1).mean()) / ext
    eF = (c["state_out"]["F"] - D["F"][t].float().reshape(-1, 3, 3)).abs().flatten(1).max(1).values
    out.append((t, dx, float(eF.max()), float(eF.mean()), float((eF > 1e-3).float().mean()), c["rec"]["pos_err_model"]))
print(" | ".join(f"프레임{t + 1} 위치차 {100 * a:.2e}% F차 최대 {b:.2e} 평균 {m:.2e} 1e-3넘는입자 {100 * f:.4f}% 오차 {100 * e:.3f}%" for t, a, b, m, f, e in out))
print("MAXDX", max(o[1] for o in out), "MEANDF", max(o[3] for o in out), "FRACDF", max(o[4] for o in out))
PY
)
  echo "[재현 시험 --net_inertial 없음] 반복 0, 프레임 0 망 = noinertial 다섯 궤적 망: 옮긴 상태 vs roll_noinertial_multi_0004.pt -- $RES | Traceback $(grep -a -c Traceback $RP/run.log)" | tee $U/repro_test.txt
  python -c "import sys;t=sys.argv[1].split();i=t.index('MAXDX');sys.exit(0 if float(t[i+1])<1e-4 and float(t[i+3])<1e-5 and float(t[i+5])<1e-3 else 1)" "$RES" || fail "재현 시험 실패 (--net_inertial 없는 옮기기가 메인 롤아웃과 다르다)"
  touch $U/repro_$C.ok; up
fi
# ---- 2) 짧은 맞춤 + 이어 돌리기 시험: A = 3 프레임 한 번에 (20 반복), B = 2 프레임 뒤 다시 띄워 3 프레임째 이어 돎 ----
if [ ! -f $U/resume_$C.ok ]; then
  g=$(wait_gpu); RT=$U/test_$C/resume; [ -e $RT ] && mv $RT ${RT}_prev_$(date +%H%M%S); mkdir -p $RT/A $RT/B
  t0=$(date +%s); $(pfcmd 0004 20 $RT/A $g | sed 's/--eval_len 80/--eval_len 3/') > $RT/A.log 2>&1 < /dev/null; t1=$(date +%s)
  $(pfcmd 0004 20 $RT/B $g | sed 's/--eval_len 80/--eval_len 2/') > $RT/B2.log 2>&1 < /dev/null
  $(pfcmd 0004 20 $RT/B $g | sed 's/--eval_len 80/--eval_len 3/') > $RT/B3.log 2>&1 < /dev/null
  RS=$(grep -a -m1 "이어 돈다" $RT/B3.log | cut -c1-120); TB=$(cat $RT/A.log $RT/B2.log $RT/B3.log | grep -a -c Traceback)
  CMP=$(python - $RT/A/frames/frame_02.pt $RT/B/frames/frame_02.pt <<'PY'
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False); b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
d = max(float((a["net"][k].float() - b["net"][k].float()).abs().max()) for k in a["net"])
s = max(float(b["net"][k].float().abs().max()) for k in b["net"])
o = max(float((x - y).abs().max()) for x, y in zip([t for st in a["opt"]["state"].values() for t in st.values() if torch.is_tensor(t) and t.numel() > 1],
                                                    [t for st in b["opt"]["state"].values() for t in st.values() if torch.is_tensor(t) and t.numel() > 1]))
dx = float((a["state_out"]["x"] - b["state_out"]["x"]).norm(dim=-1).mean()) / float(a["EXT"])
print(f"프레임 3 망 최대차 {d:.3e} (최대 크기 {s:.3e})  Adam 상태 최대차 {o:.3e}  모델 상태 위치차 {100 * dx:.2e}%  Adam 단계 {int(a['opt']['state'][0]['step'])}/{int(b['opt']['state'][0]['step'])}")
PY
)
  echo "[이어 돌리기 시험 --net_inertial 없음] B 두 번째 실행: $RS | 한 번에 vs 이어 돈 것: $CMP | Traceback $TB | 3 프레임 x 20 반복 $((t1-t0)) s" | tee $U/resume_test.txt
  grep -a "프레임별 맞춤\] 프레임" $RT/A.log | cut -c1-240 | tee -a $U/resume_test.txt
  [ -n "$RS" ] && [ "$TB" = 0 ] && grep -q "Adam 단계 60/60" <<<"$CMP" || fail "이어 돌리기 확인 실패"
  python -c "import re,sys;m=re.search(r'망 최대차 ([0-9.e+-]+)', sys.argv[1]);sys.exit(0 if m and float(m.group(1))<1e-3 else 1)" "$CMP" || fail "이어 돈 것과 한 번에 돈 것의 망 차이가 1e-3 이상"
  touch $U/resume_$C.ok; up
fi
# ---- 3) E* ----
estar(){ T=$1; g=$2; O=$U/estar_$T.json
  [ -f $U/.done_estar_$T ] && return 0
  CMD="env AF_IP_STATES=$S/roll_ov_$T.pt AF_OV_MIN_STATES=$O CUDA_VISIBLE_DEVICES=$g python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $U/estar_run_$T --tag OVM $BASE $OV --iters 0 --eval_t0 0 --eval_len 80 --tb none"
  echo "[E* 명령] $CMD" > $U/estar_cmd_$T.txt
  $CMD > $U/estar_$T.log 2>&1 < /dev/null
  N=$(python -c "import json;print(len(json.load(open('$O'))))" 2>/dev/null)
  [ "$N" = 80 ] && touch $U/.done_estar_$T || echo "[오류] E* $T 상태 $N 개"; up; }
# ---- 4) 맞춤·후처리 ----
fitrun(){ T=$1; N=$2; g=$3; O=$U/n$N/$T; mkdir -p $O
  [ -f $O/roll_pf.pt ] && grep -a -q PERFRAME_FIT_DONE $O/run_all.log 2>/dev/null && return 0
  CMD=$(pfcmd $T $N $O $g AF_PERFRAME_R2=$R/n$N/$T)
  echo "[맞춤 명령 $(date '+%F %T') 커밋 $C] $CMD" | tee -a $O/cmd.txt >> $U/PERFRAME_LAUNCHED; up
  $CMD >> $O/run_all.log 2>&1 < /dev/null
  grep -a -q PERFRAME_FIT_DONE $O/run_all.log || { echo "[오류] 맞춤 $T $N 끝나지 않음"; return 1; }
  rclone copyto $O/roll_pf.pt $R/n$N/$T/roll_pf.pt -q; }
post(){ T=$1; N=$2; g=$3; O=$U/n$N/$T
  [ -f $O/.done_post ] && return 0
  [ -f $O/h5_pf/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_pf.pt --traj $(gt_of $T) --out $O/h5_pf > $O/h5_pf.log 2>&1
  [ -f $O/pf.mp4 ] || CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $O/h5_pf --config $W/lbdata/config.json --shape lego --out $O/pf.mp4 > $O/r_pf.log 2>&1 < /dev/null
  [ -f $O/pf.mp4 ] || { echo "[오류] $T $N 렌더"; return 1; }
  D=$O/psi; mkdir -p $D
  ( cd $WP && { [ -f $D/col_pf.npy ] || CUDA_VISIBLE_DEVICES=$g python -u exe/psi_color_lb.py --gt_dir $(gt_of $T) --pt $(pt_of $T) --dump sp=$S/roll_ov_$T.pt --dump ni=$NI/roll_noinertial_multi_$T.pt --dump pf=$O/roll_pf.pt --out $D \
        --eps 1e-3 --q 1 99 --cmap turbo --range $RANGE --stats_name defstats_$T.txt --stats_frames 1 40 80 > $D/psi_color.log 2>&1 < /dev/null; }
    grep -q PSI_COLOR_DONE $D/psi_color.log || { tail -5 $D/psi_color.log; exit 1; }
    [ -f $D/h5_pf/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_pf.pt --traj $(gt_of $T) --out $D/h5_pf > $D/h5_pf.log 2>&1
    [ -f $D/psi_pf_lego.mp4 ] || CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $D/h5_pf --config $W/lbdata/config.json --shape lego \
       --colors $D/col_pf.npy --out $D/psi_pf_lego.mp4 > $D/r_pf.log 2>&1 < /dev/null
    [ -f $D/psi_pf_lego.mp4 ] ) || { echo "[오류] $T $N psi"; return 1; }
  # 정답·출력만 최적화 칸은 lego_psi_spstick, noinertial 다섯 궤적 칸은 lbours/noinertial/psi 의 같은 범위 렌더를 쓴다
  cp $NS/$T/psi_gt_lego.mp4 $D/psi_gt_lego.mp4; cp $NS/$T/psi_sp_lego.mp4 $D/psi_sp_lego.mp4; cp $NI/psi/$T/psi_ni_lego.mp4 $D/psi_ni_lego.mp4
  cp $NS/$T/range.json $D/range_spstick.json; cp $NI/psi/$T/range.json $D/range_noinertial.json
  grep -a "\[색 범위\]" $D/psi_color.log; cat $D/defstats_$T.txt
  touch $O/.done_post; up; }
traj(){ # 한 궤적: E* 와 1000 반복 맞춤을 같은 GPU 에 동시에 -> 판정(첫 궤적만) -> 필요하면 3000 -> 후처리 -> 표식
  T=$1; g=$(wait_gpu); echo "[궤적 $T] GPU$g $(date '+%F %T')" | tee -a $U/gpu.log
  ( estar $T $g >> $U/task_estar_$T.log 2>&1 ) & EP=$!
  fitrun $T 1000 $g >> $U/task_n1000_$T.log 2>&1 || fail "1000 반복 $T 실패"
  while kill -0 $EP 2>/dev/null; do sleep 60; done
  [ -f $U/.done_estar_$T ] || fail "E* $T 실패"
  if [ ! -f $U/decision.txt ]; then
    python -u exe/perframe_decide.py --u $U --n 1000 --min_frames 10 --trajs $T > $U/decide.log 2>&1 && cp $U/decide.log $U/decision.txt || fail "판정 실패"
    echo "[판정] $(tail -1 $U/decision.txt)"; up; fi
  if grep -q "^DECISION n3000" $U/decision.txt; then fitrun $T 3000 $g >> $U/task_n3000_$T.log 2>&1 || fail "3000 반복 $T 실패"; post $T 3000 $g >> $U/task_post_$T.log 2>&1 || fail "후처리 $T 3000"; fi
  post $T 1000 $g >> $U/task_post_$T.log 2>&1 || fail "후처리 $T 1000"
  python -u exe/perframe_decide.py --u $U --n 1000 --summary --trajs $T > $U/summary_$T.txt 2>&1; cat $U/summary_$T.txt
  rclone copy $U $R -q; touch $U/PERFRAME_${T}_DONE; rclone copy $U $R -q; echo "[궤적 $T 끝] $(date '+%F %T')"; }
traj 0004
traj L000
kill $SYNC 2>/dev/null
rclone copy $U $R -q && echo "[R2 업로드 완료] $(date)" && touch $U/ALL_DONE $U/PERFRAME_CLUSTER_DONE && rclone copy $U $R -q
rclone check $U $R --one-way --size-only -q && echo "[R2 대조] 전부 올라감"
echo PERFRAME_CLUSTER_DONE
