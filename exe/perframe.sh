#!/bin/bash
# 프레임별 맞춤 (사용자 10-22): "모델·증분 포텐셜 목적함수로 한 프레임은 맞출 수 있는데 여러 프레임을 함께 학습하면 간섭 때문에 안 된다" 확인.
#   궤적 T (L000, 0004) 마다 프레임 t = 0..79:
#     1) 망 갱신: 출력만 최적화 상태 t (lbours/spstick/roll_ov_T.pt) 하나만으로 학습과 같은 한 스텝 손실(phys_w·E + det 벌점)로 고정 반복 Adam
#     2) 옮기기: 그 망으로 모델 자신의 롤아웃 상태에서 한 스텝 (출력만 최적화 상태 안 씀)
#   망: 프레임 0 = 새 망(출력층 0 초기화, 입력 표준화 = 다섯 궤적 학습 체크포인트의 것), t>0 = 직전 프레임 망·Adam 에서 이어서.
#   구현: exe/train_deform.py _perframe_fit (AF_PERFRAME_FIT). BASE = spstick.sh 글자 그대로 (원문에서 다시 만들어 대조, spstick.out [BASE] 줄과 대조).
# 순서: (0) 기본 경로 대조(다섯 궤적 학습 1 반복: 이 커밋 vs 13471f4) (1) 재현 시험(반복 0 + 다섯 궤적 망 -> roll_mm 덤프와 같은 상태)
#       (2) 이어 돌리기 시험 (3) E* = 출력만 최적화 상태별 최솟값 (AF_OV_MIN_STATES, 출력만 최적화 실행 설정 그대로)
#       (4) 1000 반복/프레임 L000·0004 동시 (5) 10 프레임 뒤 판정 -> 부족하면 3000 반복/프레임도 (6) 3DGS 영상·psi 색
# 멱등: 잡이 바뀌어 다시 띄우면 끝난 단계는 건너뛰고 맞춤은 프레임 파일에서 이어 돈다.
# 출력은 처음부터 R2 lbours/perframe_fit 와 통째 동기화 (제외 없음, 5분), 프레임 파일은 생기는 즉시 (train_deform.py 가) 올린다.
# 잡 안에서만: AF_DETACH=1 bash exe/job_run.sh $W/perframe_boot.sh <로그>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
P=$W/lbpool; U=$W/lbours/perframe_fit; S=$W/lbours/spstick; NS=$W/lbours/lego_psi_spstick; WS=$W/wt/spstick_13471f4
WP=$W/wt/psi_d993fb5; RANGE=$W/lbours/lego_psi/range.json; MCK=$S/multi/ck.pt
R=r2:storage/result/anchorflow/lbours/perframe_fit; mkdir -p $U
if [ -f $U/driver.hb ] && [ $(( $(date +%s) - $(cat $U/driver.hb) )) -lt 300 ]; then echo "[중단] 다른 드라이버가 돌고 있다 (하트비트 5 분 안) $(date)"; exit 0; fi
echo $$ > $U/driver.pid
( while true; do date +%s > $U/driver.hb; sleep 120; done ) & HB=$!
( while true; do rclone copy $U $R -q; sleep 300; done ) & SYNC=$!
trap 'kill $SYNC $HB 2>/dev/null' EXIT
echo "[시작] $(date) 커밋 $C 노드 $(hostname) 잡 ${PBS_JOBID:-?}"
up(){ rclone copy $U $R -q --include "*.mp4" --include "*.txt" --include "*.json" --include "*.jsonl" --include "*.log" --include "PERFRAME_*" --max-depth 2; }
fail(){ echo "[오류] $1 $(date)"; echo "$1" > $U/PERFRAME_FAIL; up; exit 1; }
# ---- spstick.sh 와 같은 식 (원문에서 다시 만든다) ----
BASE=$(P=$P bash -c "$(grep -E '^(COMMON|REST_OLD|FLOOR_SP|REST|BASE_SP|BASE_NOXN|BASE)=' $W/spstick.sh; echo 'echo "$BASE"')")
OV=$(bash -c "$(grep -E '^OV=' $W/spstick.sh); echo \"\$OV\"")
[ -n "$BASE" ] && [ "$OV" = "--out_var --ov_init predict --ov_roll 500 --ov_opt lbfgs --ov_vars out --ov_obj ip" ] || fail "spstick.sh 에서 BASE/OV 를 못 만들었다"
[ "[BASE] $BASE" = "$(grep -a -m1 '^\[BASE\]' $W/spstick.out)" ] || fail "BASE 가 spstick.out 과 다르다"
gt_of(){ [ $1 = L000 ] && echo $W/lbdata/test_low/L000 || echo $W/lbdata/test/$1; }
pt_of(){ [ $1 = L000 ] && echo $W/lbpt/test_low/L000.pt || echo $W/lbpt/test/$1.pt; }
for T in L000 0004; do [ -f $S/roll_ov_$T.pt ] || fail "출력만 최적화 덤프 없음 $S/roll_ov_$T.pt"; [ -f $NS/roll_mm_$T.pt ] || fail "다섯 궤적 모델 덤프 없음 $NS/roll_mm_$T.pt"; done
[ -f $MCK ] || fail "다섯 궤적 학습 체크포인트 없음 $MCK"
# 명령줄: 다섯 궤적 학습 = AF_STATE_FIT_MULTI=... python -u exe/train_deform.py --infer --data $W/multi5 --hold_traj L000,0000,0001,0002,0003 --out .. --tag EV $BASE --iters 0 --eval_t0 0 --eval_len 80 --tb none
#         프레임별   = AF_PERFRAME_FIT=... python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out .. --tag PF $BASE --iters 0 --eval_t0 0 --eval_len 80 --tb none
pfcmd(){ # pfcmd <T> <반복> <출력디렉토리> <gpu> [추가 env]
  local T=$1 N=$2 O=$3 g=$4; shift 4
  echo "env $* AF_PERFRAME_FIT=$S/roll_ov_$T.pt:$N AF_PERFRAME_OUT=$O AF_PERFRAME_STATS=$MCK CUDA_VISIBLE_DEVICES=$g python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $O/run --tag PF $BASE --iters 0 --eval_t0 0 --eval_len 80 --tb none"; }
{ echo "[spstick BASE] $BASE"
  echo "[다섯 궤적 학습 명령줄 (spstick.sh multi)] AF_STATE_FIT_MULTI=\$SPEC:15000:8 AF_STATE_FIT_LOSS=ip ... python -u exe/train_deform.py --infer --data $W/multi5 --hold_traj L000,0000,0001,0002,0003 --out \$U/multi/run --tag EV $BASE --iters 0 --eval_t0 0 --eval_len 80 --tb none"
  echo "[프레임별 맞춤 명령줄 L000 1000] $(pfcmd L000 1000 $U/n1000/L000 X AF_PERFRAME_R2=$R/n1000/L000)"
  X=$(pfcmd L000 1000 $U/n1000/L000 X | sed 's/.* python -u exe\/train_deform.py //')
  OUTD=$(comm -23 <(tr ' ' '\n' <<<"--infer --data $W/multi5 --hold_traj L000,0000,0001,0002,0003 --out RUN --tag EV $BASE --iters 0 --eval_t0 0 --eval_len 80 --tb none" | sort) <(tr ' ' '\n' <<<"$X" | sed "s#$U/n1000/L000/run#RUN#" | sort) | paste -sd' ')
  IND=$(comm -13 <(tr ' ' '\n' <<<"--infer --data $W/multi5 --hold_traj L000,0000,0001,0002,0003 --out RUN --tag EV $BASE --iters 0 --eval_t0 0 --eval_len 80 --tb none" | sort) <(tr ' ' '\n' <<<"$X" | sed "s#$U/n1000/L000/run#RUN#" | sort) | paste -sd' ')
  echo "[명령줄 차이 다섯 궤적 학습 -> 프레임별 (python 인자)] 빠짐: '$OUTD' | 더함: '$IND'"
  echo "[BASE 포함 확인] 프레임별 명령줄에 spstick BASE 가 글자 그대로 $(grep -c -F -- " $BASE " <<<"$(pfcmd L000 1000 $U/n1000/L000 X)") 번"; } | tee $U/cmdline_diff.txt
[ "$(grep -c -F -- " $BASE " <<<"$(pfcmd L000 1000 $U/n1000/L000 X)")" = 1 ] || fail "프레임별 명령줄에 BASE 가 글자 그대로 없다"
free_gpu(){ # 빈 GPU (메모리 1GB 미만 + 프로세스 없음), GPU 0 은 절대 안 씀 (noinertial 학습), 인자 카드들도 뺀다
  for c in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<1024{print $1}'); do
    [ "$c" = 0 ] && continue
    case " $* " in *" $c "*) continue ;; esac
    u=$(nvidia-smi -i $c --query-gpu=uuid --format=csv,noheader)
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u" && continue
    echo $c; return; done; }
wait_gpu(){ local g=""; while [ -z "$g" ]; do g=$(free_gpu "$@"); [ -z "$g" ] && { echo "[대기] 빈 GPU 없음 $(date +%T)" >> $U/wait.log; sleep 120; }; done
  echo "[GPU 배정] $g $(date '+%F %T') $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $U/gpu.log; echo $g; }
# ---- 0) 기본 경로 대조: 다섯 궤적 학습 1 반복 -- 이 커밋 vs spstick 학습 커밋 13471f4 ----
SPEC=""; for T in L000 0000 0001 0002 0003; do SPEC="$SPEC${SPEC:+,}$T=$S/roll_ov_$T.pt"; done
if [ ! -f $U/base_equiv_$C.ok ]; then
  g=$(wait_gpu); echo "[기본 경로 대조] GPU$g $(date +%T)"
  BE=$U/base_equiv_$C; [ -e $BE ] && mv $BE ${BE}_prev_$(date +%H%M%S); mkdir -p $BE/new $BE/old
  mf(){ env AF_STATE_FIT_MULTI=$SPEC:1:8 AF_STATE_FIT_LOSS=ip AF_STATE_FIT_EVERY=1 AF_STATE_FIT_EVAL=100000 AF_STATE_FIT_CK=$1/ck.pt AF_STATE_FIT_TB=$1/tb \
    CUDA_VISIBLE_DEVICES=$g python -u exe/train_deform.py --infer --data $W/multi5 --hold_traj L000,0000,0001,0002,0003 --out $1/run --tag EV $BASE --iters 0 --eval_t0 0 --eval_len 1 --tb none > $1.log 2>&1 < /dev/null; }
  mf $BE/new; ( cd $WS && mf $BE/old )
  CMPB=$(python - $BE/new/ck.pt $BE/old/ck.pt <<'PY'
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False); b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
st = {k: float((a["net"][k].float() - b["net"][k].float()).abs().max()) for k in a["net"] if k in b["net"]}
print(f"입력폭 {a.get('n_feat')}/{b.get('n_feat')}  키 같음 {sorted(a['net']) == sorted(b['net'])}  1 반복 뒤 망 최대차 {max(st.values()):.3e}")
PY
)
  L1=$(grep -a -m1 "\[다궤적 적합\] 1/1 " $BE/new.log | grep -o "손실 [0-9.e+-]*"); L0=$(grep -a -m1 "\[다궤적 적합\] 1/1 " $BE/old.log | grep -o "손실 [0-9.e+-]*")
  TB=$(cat $BE/new.log $BE/old.log | grep -a -c Traceback)
  echo "[기본 경로 대조] 이 커밋 $C 첫 반복 $L1 | 13471f4 첫 반복 $L0 | $CMPB | Traceback $TB" | tee $U/base_equiv.txt
  [ -n "$L1" ] && [ "$L1" = "$L0" ] && grep -q "입력폭 33/33  키 같음 True" <<<"$CMPB" && [ "$TB" = 0 ] || fail "기본 경로가 13471f4 와 다르다"
  python -c "import re,sys;m=re.search(r'망 최대차 ([0-9.e+-]+)', sys.argv[1]);sys.exit(0 if m and float(m.group(1))<1e-5 else 1)" "$CMPB" || fail "기본 경로 망 차이 1e-5 이상"
  touch $U/base_equiv_$C.ok; up
fi
# ---- 1) 재현 시험: 반복 0 + 프레임 0 망 = 다섯 궤적 학습 망 -> 옮긴 상태가 lego_psi_spstick/roll_mm_L000.pt (같은 망 롤아웃) 와 같아야 ----
if [ ! -f $U/repro_$C.ok ]; then
  g=$(wait_gpu); RP=$U/test_$C/repro; [ -e $RP ] && mv $RP ${RP}_prev_$(date +%H%M%S); mkdir -p $RP
  CMD=$(pfcmd L000 0 $RP $g AF_PERFRAME_INIT=$MCK | sed 's/--eval_len 80/--eval_len 6/'); echo "[재현 시험 명령] $CMD" | tee $RP/cmd.txt
  $CMD > $RP/run.log 2>&1 < /dev/null
  RES=$(python - $RP/frames $NS/roll_mm_L000.pt <<'PY'
import sys, os, torch
D = torch.load(sys.argv[2], map_location="cpu", weights_only=False); ext = float(D["EXT"])
out = []
for t in range(6):
    f = os.path.join(sys.argv[1], f"frame_{t:02d}.pt")
    c = torch.load(f, map_location="cpu", weights_only=False)
    dx = float((c["state_out"]["x"] - D["pred"][t].float()).norm(dim=-1).mean()) / ext
    dF = float((c["state_out"]["F"] - D["F"][t].float().reshape(-1, 3, 3)).abs().max())
    out.append((t, dx, dF, c["rec"]["pos_err_model"]))
print(" | ".join(f"프레임{t + 1} 위치차 {100 * a:.2e}% F최대차 {b:.2e} 오차 {100 * e:.3f}%" for t, a, b, e in out))
print("MAXDX", max(a for _, a, _, _ in out), "MAXDF", max(b for _, _, b, _ in out))
PY
)
  echo "[재현 시험] 반복 0, 프레임 0 망 = 다섯 궤적 학습 망: 옮긴 모델 상태 vs roll_mm_L000.pt -- $RES | Traceback $(grep -a -c Traceback $RP/run.log)" | tee $U/repro_test.txt
  python -c "import sys;t=sys.argv[1].split();i=t.index('MAXDX');sys.exit(0 if float(t[i+1])<1e-4 and float(t[i+3])<1e-3 else 1)" "$RES" || fail "재현 시험 실패 (옮기기가 메인 롤아웃과 다르다)"
  touch $U/repro_$C.ok; up
fi
# ---- 2) 이어 돌리기 시험: A = 3 프레임 한 번에 (20 반복), B = 2 프레임 뒤 다시 띄워 3 프레임째 이어 돎 -> 비교 ----
if [ ! -f $U/resume_$C.ok ]; then
  g=$(wait_gpu); RT=$U/test_$C/resume; [ -e $RT ] && mv $RT ${RT}_prev_$(date +%H%M%S); mkdir -p $RT/A $RT/B
  t0=$(date +%s); $(pfcmd L000 20 $RT/A $g | sed 's/--eval_len 80/--eval_len 3/') > $RT/A.log 2>&1 < /dev/null; t1=$(date +%s)
  $(pfcmd L000 20 $RT/B $g | sed 's/--eval_len 80/--eval_len 2/') > $RT/B2.log 2>&1 < /dev/null
  $(pfcmd L000 20 $RT/B $g | sed 's/--eval_len 80/--eval_len 3/') > $RT/B3.log 2>&1 < /dev/null
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
  echo "[이어 돌리기 시험] B 두 번째 실행: $RS | 한 번에 vs 이어 돈 것: $CMP | Traceback $TB | 3 프레임 x 20 반복 $((t1-t0)) s" | tee $U/resume_test.txt
  [ -n "$RS" ] && [ "$TB" = 0 ] && grep -q "Adam 단계 60/60" <<<"$CMP" || fail "이어 돌리기 확인 실패"
  python -c "import re,sys;m=re.search(r'망 최대차 ([0-9.e+-]+)', sys.argv[1]);sys.exit(0 if m and float(m.group(1))<1e-3 else 1)" "$CMP" || fail "이어 돈 것과 한 번에 돈 것의 망 차이가 1e-3 이상"
  grep -a "프레임별 맞춤\] 프레임" $RT/A.log | cut -c1-220 | tee -a $U/resume_test.txt
  touch $U/resume_$C.ok; up
fi
# ---- 3) E* (출력만 최적화 상태별 최솟값): 출력만 최적화 실행(spstick.sh ov)과 같은 BASE $OV 로 상태마다 한 스텝 L-BFGS ----
estar(){ T=$1; g=$2; O=$U/estar_$T.json
  [ -f $U/.done_estar_$T ] && return 0
  CMD="env AF_IP_STATES=$S/roll_ov_$T.pt AF_OV_MIN_STATES=$O CUDA_VISIBLE_DEVICES=$g python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $U/estar_run_$T --tag OVM $BASE $OV --iters 0 --eval_t0 0 --eval_len 80 --tb none"
  echo "[E* 명령] $CMD" > $U/estar_cmd_$T.txt
  $CMD > $U/estar_$T.log 2>&1 < /dev/null
  N=$(python -c "import json;print(len(json.load(open('$O'))))" 2>/dev/null)
  [ "$N" = 80 ] && touch $U/.done_estar_$T || echo "[오류] E* $T 상태 $N 개"; up; }
# ---- 4·5) 맞춤 ----
fitrun(){ T=$1; N=$2; g=$3; O=$U/n$N/$T; mkdir -p $O
  [ -f $O/roll_pf.pt ] && grep -a -q PERFRAME_FIT_DONE $O/run_all.log 2>/dev/null && return 0
  CMD=$(pfcmd $T $N $O $g AF_PERFRAME_R2=$R/n$N/$T)
  echo "[맞춤 명령 $(date '+%F %T') 커밋 $C] $CMD" | tee -a $O/cmd.txt >> $U/PERFRAME_LAUNCHED
  $CMD >> $O/run_all.log 2>&1 < /dev/null
  grep -a -q PERFRAME_FIT_DONE $O/run_all.log || { echo "[오류] 맞춤 $T $N 끝나지 않음"; return 1; }
  rclone copyto $O/roll_pf.pt $R/n$N/$T/roll_pf.pt -q; }
post(){ T=$1; N=$2; g=$3; O=$U/n$N/$T
  [ -f $O/.done_post ] && return 0
  [ -f $O/h5_pf/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_pf.pt --traj $(gt_of $T) --out $O/h5_pf > $O/h5_pf.log 2>&1
  [ -f $O/pf.mp4 ] || CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $O/h5_pf --config $W/lbdata/config.json --shape lego --out $O/pf.mp4 > $O/r_pf.log 2>&1 < /dev/null
  [ -f $O/pf.mp4 ] || { echo "[오류] $T $N 렌더"; return 1; }
  D=$O/psi; mkdir -p $D
  ( cd $WP && { [ -f $D/col_pf.npy ] || CUDA_VISIBLE_DEVICES=$g python -u exe/psi_color_lb.py --gt_dir $(gt_of $T) --pt $(pt_of $T) --dump sp=$S/roll_ov_$T.pt --dump mm=$NS/roll_mm_$T.pt --dump pf=$O/roll_pf.pt --out $D \
        --eps 1e-3 --q 1 99 --cmap turbo --range $RANGE --stats_name defstats_$T.txt --stats_frames 1 40 80 > $D/psi_color.log 2>&1 < /dev/null; }
    grep -q PSI_COLOR_DONE $D/psi_color.log || { tail -5 $D/psi_color.log; exit 1; }
    [ -f $D/h5_pf/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_pf.pt --traj $(gt_of $T) --out $D/h5_pf > $D/h5_pf.log 2>&1
    [ -f $D/psi_pf_lego.mp4 ] || CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $D/h5_pf --config $W/lbdata/config.json --shape lego \
       --colors $D/col_pf.npy --out $D/psi_pf_lego.mp4 > $D/r_pf.log 2>&1 < /dev/null
    [ -f $D/psi_pf_lego.mp4 ] ) || { echo "[오류] $T $N psi"; return 1; }
  for n in gt sp mm; do cp $NS/$T/psi_${n}_lego.mp4 $D/psi_${n}_lego.mp4; done; cp $NS/$T/range.json $D/range_spstick.json
  grep -a "\[색 범위\]" $D/psi_color.log; cat $D/defstats_$T.txt
  touch $O/.done_post; up; }
job(){ T=$1; N=$2; g=$3; echo "[시작] $T n$N GPU$g $(date +%T)"; fitrun $T $N $g && post $T $N $g && touch $U/.done_n${N}_$T; echo "[끝] $T n$N GPU$g $(date +%T)"; }
declare -A PIDS
G1=$(wait_gpu); ( estar L000 $G1 >> $U/task_estar_L000.log 2>&1 ) & PIDS[eL]=$!; sleep 60
G2=$(wait_gpu $G1); ( estar 0004 $G2 >> $U/task_estar_0004.log 2>&1 ) & PIDS[e4]=$!; sleep 60
G3=$(wait_gpu $G1 $G2); ( job L000 1000 $G3 >> $U/task_n1000_L000.log 2>&1 ) & PIDS[aL]=$!; sleep 60
G4=$(wait_gpu $G1 $G2 $G3); ( job 0004 1000 $G4 >> $U/task_n1000_0004.log 2>&1 ) & PIDS[a4]=$!
echo "[배정] E* L000 GPU$G1, E* 0004 GPU$G2, 1000 L000 GPU$G3, 1000 0004 GPU$G4 $(date '+%F %T')" | tee -a $U/gpu.log; up
# 판정: 두 궤적 모두 10 프레임 이상 + 그 프레임의 E* -> exe/perframe_decide.py
DEC=""
while :; do
  sleep 300
  if [ -z "$DEC" ] && [ ! -f $U/decision.txt ]; then
    python -u exe/perframe_decide.py --u $U --n 1000 --min_frames 10 > $U/decide.log 2>&1 && DEC=$(tail -1 $U/decide.log)
    [ -n "$DEC" ] && { cp $U/decide.log $U/decision.txt; echo "[판정] $DEC $(date +%T)"; up; }
  elif [ -z "$DEC" ]; then DEC=$(tail -1 $U/decision.txt); fi
  if grep -q "^DECISION n3000" <<<"$DEC" && [ -z "${PIDS[bL]:-}" ]; then
    g=$(wait_gpu); ( job L000 3000 $g >> $U/task_n3000_L000.log 2>&1 ) & PIDS[bL]=$!; sleep 60
    g2=$(wait_gpu $g); ( job 0004 3000 $g2 >> $U/task_n3000_0004.log 2>&1 ) & PIDS[b4]=$!
    echo "[배정] 3000 L000 GPU$g, 3000 0004 GPU$g2 $(date '+%F %T')" | tee -a $U/gpu.log; up
  fi
  run=0; for k in "${!PIDS[@]}"; do kill -0 ${PIDS[$k]} 2>/dev/null && run=1; done
  [ $run = 0 ] && { [ -n "$DEC" ] || { python -u exe/perframe_decide.py --u $U --n 1000 --min_frames 10 > $U/decide.log 2>&1; DEC=$(tail -1 $U/decide.log); cp $U/decide.log $U/decision.txt; }
                    grep -q "^DECISION n3000" <<<"$DEC" && [ -z "${PIDS[bL]:-}" ] && continue; break; }
done
for T in L000 0004; do [ -f $U/.done_estar_$T ] || fail "E* $T 실패"; [ -f $U/.done_n1000_$T ] || fail "1000 반복 $T 실패"; done
grep -q "^DECISION n3000" $U/decision.txt && for T in L000 0004; do [ -f $U/.done_n3000_$T ] || fail "3000 반복 $T 실패"; done
python -u exe/perframe_decide.py --u $U --n 1000 --min_frames 80 --summary > $U/summary.txt 2>&1; cat $U/summary.txt
kill $SYNC 2>/dev/null
rclone copy $U $R -q && echo "[R2 업로드 완료] $(date)" && touch $U/ALL_DONE $U/PERFRAME_CLUSTER_DONE && rclone copy $U $R -q
rclone check $U $R --one-way --size-only -q && echo "[R2 대조] 전부 올라감"
echo PERFRAME_CLUSTER_DONE
