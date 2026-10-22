#!/bin/bash
# 목적함수 셋 비교 (사용자 10-22, pf3obj): "망도 목적함수도 문제가 아니라 증분 포텐셜을 망 파라미터로 최적화하는 조합이 어렵다" 를 지금 설정에서 확인.
#   방식 = 예전 B3i/B3j (--ov_vars net, ovnet_t1.sh adam3: Adam 1e-3, 2000 회/프레임) 그대로: 매 프레임 모델 자신의 롤아웃 상태에서 망 파라미터를
#   그 프레임 목적함수로 내리고 -> 그 망 출력으로 한 스텝 -> 다음 프레임 (망은 이어 씀, 새 망 = 출력층 0 초기화).
#   설정 = spstick.sh BASE 글자 그대로 (새 바닥 다섯 궤적 학습과 같은 BASE). 목적함수만 셋:
#     ip   = 증분 포텐셜 (--ov_obj ip)
#     l2ov = 출력만 최적화 덤프(lbours/spstick/roll_ov_T.pt)의 같은 프레임 번호 다음 상태 위치로 L2 (--ov_obj l2 + AF_OV_L2_TGT)
#     l2gt = 정답 다음 프레임 L2 (--ov_obj l2, 기존 그대로)
#     nt   = 뉴턴 목표 L2 (--ov_obj newton, 사용자 10-22 추가): 망 노드 변위 u 에서 증분 포텐셜의 감쇠 뉴턴 걸음(CG 최대 50, 음곡률 직전까지)으로
#            목표 u_t = sg(u + d), 손실 Σ M|u - u_t|²/ΣM; 목표는 NEV 반복마다 다시
#   기록 (exe/train_deform.py _ovnet_fit = AF_OVNET_FIT): 프레임별 목적함수 곡선, 같은 출력의 증분 포텐셜 E, 같은 모델 상태에서 출력만 최적화로 푼 E*
#   (spstick OV 플래그로 같은 _ov_frame 경로, L-BFGS 500), 위치 오차, 망 파라미터·모델 상태 frames/frame_XX.pt
# 단계 PF3_STAGE=test: 명령줄 대조 -> (1) 메인 롤아웃 경로와 같은지 (2) 이어 돌리기 (3) E* 경로 = 출력만 최적화 실행 (4) 짧은 시험(세 목적함수 L000 2 프레임)
#       PF3_STAGE=full: 시험 표식 확인 -> 6 실행(L000·0004 x 3) 빈 GPU 에 나눠 동시에 -> 실행마다 3DGS 렌더 -> 궤적마다 psi 색 -> 요약
# 멱등: 다시 띄우면 끝난 단계는 건너뛰고 실행은 프레임 파일에서 이어 돈다. 출력은 R2 lbours/pf3obj 와 통째 동기화 (제외 없음, 5분).
# 잡 안에서만: AF_DETACH=1 bash exe/job_run.sh $W/pf3obj_boot.sh (또는 pf3obj_test_boot.sh) <로그>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD); STAGE=${PF3_STAGE:?PF3_STAGE=test 또는 full}
U=$W/lbours/pf3obj; S=$W/lbours/spstick; NS=$W/lbours/lego_psi_spstick
WP=$W/wt/psi_d993fb5; RANGE=$W/lbours/lego_psi/range.json
R=r2:storage/result/anchorflow/lbours/pf3obj; mkdir -p $U
NEV=${PF3_NEWTON_EVERY:-50}   # 뉴턴 목표를 다시 만드는 간격 (반복) -- 짧은 시험의 풀이 시간으로 정한다
if [ -f $U/driver.hb ] && [ $(( $(date +%s) - $(cat $U/driver.hb) )) -lt 300 ]; then echo "[중단] 다른 드라이버가 돌고 있다 (하트비트 5 분 안) $(date)"; exit 0; fi
echo $$ > $U/driver.pid; date +%s > $U/driver.hb
( while true; do date +%s > $U/driver.hb; sleep 120; done ) & HB=$!
( while true; do sleep 300; rclone copy $U $R -q; done ) & SYNC=$!
trap 'kill $SYNC $HB 2>/dev/null' EXIT
echo "[시작] $(date) 단계 $STAGE 커밋 $C 노드 $(hostname) 잡 ${PBS_JOBID:-?}"
up(){ rclone copy $U $R -q --include "*.mp4" --include "*.txt" --include "*.json" --include "*.log" --include "PF3OBJ_*" --max-depth 3; }
fail(){ echo "[오류] $1 $(date)"; echo "$1 ($(date '+%F %T'), 커밋 $C)" > $U/PF3OBJ_FAIL; up; exit 1; }
# ---- 명령줄: spstick BASE 원문에서 다시 만들고, 예전 adam3 실행(ovnet_t1.sh, lb_vid_lowmid.sh BASE) 과 자동 대조 ----
BASE=$(P=$W/lbpool bash -c "$(grep -E '^(COMMON|REST_OLD|FLOOR_SP|REST|BASE_SP|BASE_NOXN|BASE)=' $W/spstick.sh; echo 'echo "$BASE"')")
OV=$(bash -c "$(grep -E '^OV=' $W/spstick.sh); echo \"\$OV\"")
[ -n "$BASE" ] && [ "$OV" = "--out_var --ov_init predict --ov_roll 500 --ov_opt lbfgs --ov_vars out --ov_obj ip" ] || fail "spstick.sh 에서 BASE/OV 를 못 만들었다"
[ "[BASE] $BASE" = "$(grep -a -m1 '^\[BASE\]' $W/spstick.out)" ] || fail "BASE 가 spstick.out 과 다르다"
BASE_OLD=$(P=$W/lbpool bash -c "$(grep '^BASE=' $W/lb_vid_lowmid.sh); echo \"\$BASE\"")
XA=$(grep -o 'adam3) X="[^"]*"' $W/ovnet_t1.sh | sed 's/^adam3) X="//; s/"$//')
[ "$XA" = "--ov_vars net --ov_opt adam --out_var_lr 1e-3 --ov_roll 2000" ] && [ -n "$BASE_OLD" ] || fail "예전 ovnet_t1.sh adam3 / lb_vid_lowmid.sh BASE 를 못 읽었다 ($XA)"
gt_of(){ [ $1 = L000 ] && echo $W/lbdata/test_low/L000 || echo $W/lbdata/test/$1; }
pt_of(){ [ $1 = L000 ] && echo $W/lbpt/test_low/L000.pt || echo $W/lbpt/test/$1.pt; }
for T in L000 0004; do [ -f $S/roll_ov_$T.pt ] || fail "출력만 최적화 덤프 없음 $S/roll_ov_$T.pt"; done
objx(){ case $1 in ip) echo "--ov_obj ip";; nt) echo "--ov_obj newton";; l2ov|l2gt) echo "--ov_obj l2";; esac; }
pyargs(){ # pyargs <T> <obj> <출력>
  echo "--infer --data $W/one/$1 --hold_traj $1 --out $3/run --tag OVN $BASE --out_var --ov_init predict $XA $(objx $2) --iters 0 --eval_t0 0 --eval_len 80 --tb none"; }
fitcmd(){ # fitcmd <T> <obj> <출력> <gpu> [추가 env]
  local T=$1 K=$2 O=$3 g=$4; shift 4
  local TG=""; [ $K = l2ov ] && TG="AF_OV_L2_TGT=$S/roll_ov_$T.pt"; [ $K = nt ] && TG="AF_OV_NEWTON_EVERY=$NEV AF_OV_NEWTON_CG=50 AF_OV_NEWTON_DAMP=1e-3"
  echo "env $* $TG AF_OV_CURVE=$O/curve_unused.json AF_OVNET_FIT=$O AF_OVNET_REF=$S/roll_ov_$T.pt AF_OVNET_ESTAR=500 CUDA_VISIBLE_DEVICES=$g python -u exe/train_deform.py $(pyargs $T $K $O)"; }
OLDA(){ echo "--infer --data $W/lbours/vid/d_$1 --hold_traj $1 --out RUN --tag OVN $BASE_OLD --out_var --ov_init predict $XA --iters 0 --eval_t0 0 --eval_len 80 --tb none"; }
tokdiff(){ comm -$1 <(tr ' ' '\n' <<<"$2" | sort) <(tr ' ' '\n' <<<"$3" | sort) | paste -sd' '; }
{ echo "[spstick BASE] $BASE"
  echo "[예전 BASE (lb_vid_lowmid.sh, B3i/B3j)] $BASE_OLD"
  echo "[예전 adam3 변수·최적화기 (ovnet_t1.sh)] $XA"
  echo "[예전 python 인자 (adam3, L000)] $(OLDA L000)"
  echo "[BASE 차이 예전->지금] 빠짐: '$(tokdiff 23 "$BASE_OLD" "$BASE")' | 더함: '$(tokdiff 13 "$BASE_OLD" "$BASE")'"
  for K in ip nt l2ov l2gt; do for T in L000 0004; do
    N=$(pyargs $T $K $U/$K/$T | sed "s#$U/$K/$T/run#RUN#")
    NB=$(sed "s# $BASE # #" <<<"$N"); OB=$(sed "s# $BASE_OLD # #" <<<"$(OLDA $T)")
    echo "[명령 $K $T] $(fitcmd $T $K $U/$K/$T G AF_OVNET_R2=$R/$K/$T)"
    echo "[대조 $K $T: BASE 뺀 나머지 예전->지금] 빠짐: '$(tokdiff 23 "$OB" "$NB")' | 더함: '$(tokdiff 13 "$OB" "$NB")'  | BASE 글자 그대로 $(grep -c -F -- " $BASE " <<<"$N") 번"
  done; done; } > $U/cmdline_diff.txt
# 자동 판정: BASE 뺀 나머지 차이는 데이터 경로(spstick 실행과 같은 one/T)와 --ov_obj 뿐이어야
for K in ip nt l2ov l2gt; do for T in L000 0004; do
  N=$(pyargs $T $K $U/$K/$T | sed "s#$U/$K/$T/run#RUN#"); NB=$(sed "s# $BASE # #" <<<"$N"); OB=$(sed "s# $BASE_OLD # #" <<<"$(OLDA $T)")
  [ "$(grep -c -F -- " $BASE " <<<"$N")" = 1 ] || fail "$K $T 명령줄에 BASE 가 글자 그대로 없다"
  [ "$(tokdiff 23 "$OB" "$NB")" = "$W/lbours/vid/d_$T" ] || fail "$K $T 예전 대비 빠진 인자가 데이터 경로 말고도 있다"
  EXP=$(tr ' ' '\n' <<<"$W/one/$T $(objx $K)" | sort | paste -sd' ')
  [ "$(tokdiff 13 "$OB" "$NB")" = "$EXP" ] || fail "$K $T 예전 대비 더한 인자가 데이터 경로·--ov_obj 말고도 있다"
done; done
echo "[명령줄 대조 통과] BASE 뺀 차이 = 데이터 경로(spstick 과 같은 one/T) + --ov_obj 뿐; 환경변수 차이 = AF_OVNET_*(기록), l2ov 만 AF_OV_L2_TGT, nt 만 AF_OV_NEWTON_*" | tee -a $U/cmdline_diff.txt
free_gpu(){ # 빈 GPU (메모리 1GB 미만 + 프로세스 없음), GPU 0 은 절대 안 씀 (noinertial 학습), 인자 카드들도 뺀다
  for c in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<1024{print $1}'); do
    [ "$c" = 0 ] && continue
    case " $* " in *" $c "*) continue ;; esac
    u=$(nvidia-smi -i $c --query-gpu=uuid --format=csv,noheader)
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u" && continue
    echo $c; return; done; }
wait_gpu(){ local g=""; while [ -z "$g" ]; do g=$(free_gpu "$@"); [ -z "$g" ] && { echo "[대기] 빈 GPU 없음 $(date +%T)" >> $U/wait.log; sleep 120; }; done
  echo "[GPU 배정] $g $(date '+%F %T') $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $U/gpu.log; echo $g; }
alive(){ local r=0; for p_ in "$@"; do kill -0 $p_ 2>/dev/null && r=1; done; echo $r; }   # 끝없는 동기화 루프는 넣지 않는다 (wait 금지)

if [ $STAGE = test ]; then
  TD=$U/test_$C; mkdir -p $TD
  # (1) 메인 롤아웃 경로와 같은지: 같은 인자(반복 50, 2 프레임)로 메인 --ov_roll 분기(AF_ROLL_DUMP) vs _ovnet_fit (E* 켠 채) -> 위치·F
  # (2) 이어 돌리기: A = 3 프레임 한 번에 (반복 20), B = 2 프레임 뒤 다시 띄워 3 프레임째 -> 프레임 3 망·상태
  # (3) E* 경로: 프레임 0 (정답 시작 상태) 의 E* 해 = 출력만 최적화 실행 roll_ov_L000.pt 프레임 1
  # (4) 짧은 시험: 세 목적함수 L000 2 프레임, 2000 회 -> 프레임당 시간·목적함수 감소·ip 는 E = 목적함수
  g1=$(wait_gpu); g2=$(wait_gpu $g1); g3=$(wait_gpu $g1 $g2); g4=$(wait_gpu $g1 $g2 $g3); g5=$(wait_gpu $g1 $g2 $g3 $g4); g6=$(wait_gpu $g1 $g2 $g3 $g4 $g5)
  echo "[시험 GPU] 대조 $g1, 이어돌기 $g2, 짧은 시험 ip $g3 l2ov $g4 l2gt $g5 nt $g6" | tee -a $U/gpu.log
  for K in ip nt l2ov l2gt; do mkdir -p $TD/short_$K; done
  ( CMD=$(fitcmd L000 nt $TD/short_nt $g6 | sed 's/--eval_len 80/--eval_len 2/'); echo "$CMD" > $TD/short_nt/cmd.txt; $CMD > $TD/short_nt/run.log 2>&1 < /dev/null ) & P6=$!
  ( for K in ip; do CMD=$(fitcmd L000 $K $TD/short_$K $g3 | sed 's/--eval_len 80/--eval_len 2/'); echo "$CMD" > $TD/short_$K/cmd.txt; $CMD > $TD/short_$K/run.log 2>&1 < /dev/null; done ) & P3=$!
  ( CMD=$(fitcmd L000 l2ov $TD/short_l2ov $g4 | sed 's/--eval_len 80/--eval_len 2/'); echo "$CMD" > $TD/short_l2ov/cmd.txt; $CMD > $TD/short_l2ov/run.log 2>&1 < /dev/null ) & P4=$!
  ( CMD=$(fitcmd L000 l2gt $TD/short_l2gt $g5 | sed 's/--eval_len 80/--eval_len 2/'); echo "$CMD" > $TD/short_l2gt/cmd.txt; $CMD > $TD/short_l2gt/run.log 2>&1 < /dev/null ) & P5=$!
  ( mkdir -p $TD/eq/fit $TD/eq/main
    $(fitcmd L000 ip $TD/eq/fit $g1 | sed 's/--eval_len 80/--eval_len 2/; s/--ov_roll 2000/--ov_roll 50/') > $TD/eq/fit.log 2>&1 < /dev/null
    env AF_ROLL_DUMP=$TD/eq/main/roll.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=L000 AF_ROLL_T0=0 CUDA_VISIBLE_DEVICES=$g1 python -u exe/train_deform.py \
      $(pyargs L000 ip $TD/eq/main | sed 's/--eval_len 80/--eval_len 2/; s/--ov_roll 2000/--ov_roll 50/') > $TD/eq/main.log 2>&1 < /dev/null ) & P1=$!
  ( mkdir -p $TD/rs/A $TD/rs/B
    $(fitcmd L000 ip $TD/rs/A $g2 | sed 's/--eval_len 80/--eval_len 3/; s/--ov_roll 2000/--ov_roll 20/') > $TD/rs/A.log 2>&1 < /dev/null
    $(fitcmd L000 ip $TD/rs/B $g2 | sed 's/--eval_len 80/--eval_len 2/; s/--ov_roll 2000/--ov_roll 20/') > $TD/rs/B2.log 2>&1 < /dev/null
    $(fitcmd L000 ip $TD/rs/B $g2 | sed 's/--eval_len 80/--eval_len 3/; s/--ov_roll 2000/--ov_roll 20/') > $TD/rs/B3.log 2>&1 < /dev/null ) & P2=$!
  while [ "$(alive $P1 $P2 $P3 $P4 $P5 $P6)" = 1 ]; do sleep 60; done
  TB=$(cat $TD/eq/*.log $TD/rs/*.log $TD/short_*/run.log | grep -a -c Traceback)
  RES=$(python - $TD $S/roll_ov_L000.pt <<'PY'
import sys, os, json, torch
td, ovp = sys.argv[1:]
ld = lambda p: torch.load(p, map_location="cpu", weights_only=False)
out, ok = [], True
# (1) 메인 경로
M = ld(f"{td}/eq/main/roll.pt"); ext = float(M["EXT"])
for t in range(2):
    c = ld(f"{td}/eq/fit/frames/frame_{t:02d}.pt")
    dx = float((c["state_out"]["x"] - M["pred"][t].float()).norm(dim=-1).mean()) / ext
    dF = float((c["state_out"]["F"].reshape(-1, 9) - M["F"][t].float().reshape(-1, 9)).abs().mean())
    out.append(f"[메인 경로 대조] 프레임 {t + 1}: 위치차 {100 * dx:.2e}% (EXT 대비) F 평균차 {dF:.2e}")
    ok &= dx < 1e-5 and dF < 1e-5
# (2) 이어 돌리기
a_, b_ = ld(f"{td}/rs/A/frames/frame_02.pt"), ld(f"{td}/rs/B/frames/frame_02.pt")
dn = max(float((a_["net"][k].float() - b_["net"][k].float()).abs().max()) for k in a_["net"])
sn = max(float(b_["net"][k].float().abs().max()) for k in b_["net"])
dx = float((a_["state_out"]["x"] - b_["state_out"]["x"]).norm(dim=-1).mean()) / ext
out.append(f"[이어 돌리기] 프레임 3 망 최대차 {dn:.2e} (망 최대 크기 {sn:.2e}) 모델 위치차 {100 * dx:.2e}%")
ok &= dn < 1e-3 * max(sn, 1e-12) and dx < 1e-5
# (3) E* 경로: 프레임 0 E* 해 vs 출력만 최적화 실행 프레임 1
O = ld(ovp); c0 = ld(f"{td}/eq/fit/frames/frame_00.pt")
ds = float((c0["x_star"] - O["pred"][0].float()).norm(dim=-1).mean()) / ext
out.append(f"[E* 경로] 프레임 0 E* 해 vs spstick 출력만 최적화 프레임 1: 위치차 {100 * ds:.2e}% (E* {c0['rec']['E_star']:.6e})")
ok &= ds < 1e-4
# (4) 짧은 시험
R_ = {}
for k in ("ip", "nt", "l2ov", "l2gt"):
    r = json.load(open(f"{td}/short_{k}/records.json")); R_[k] = r
    for q in r:
        dec = q["obj_last"] < q["obj_first"]
        cv = q["obj_curve"]; mid = [e for i, e in cv if i in (0, 100, 500, 1000, 1500, 2000)]
        out.append(f"[짧은 시험 {k}] 프레임 {q['t'] + 1}: 목적함수 {q['obj_first']:.4e} -> {q['obj_last']:.4e} ({'감소' if dec else '감소 안 함'}; 반복 0/100/500/1000/1500/2000: "
                   + " ".join(f"{e:.3e}" for e in mid) + f")  E {q['E_ip']:.4e} E* {q['E_star']:.4e} E/E* {q['ratio_E_Estar']:.3f}  위치오차 {100 * q['pos_err_model']:.3f}%"
                   + f" (출력만 최적화 {100 * q['ref_err']:.3f}%)  맞춤 {q['sec_fit']:.0f} s 프레임 {q['sec']:.0f} s"
                   + (f"  뉴턴: 풀이 {q['newton']['n_solves']} 회 (매 {q['newton']['every']} 반복) CG 평균 {q['newton']['cg_mean']:.1f} 최대 {q['newton']['cg_max']} 음곡률 {q['newton']['neg_count']} 풀이당 {q['newton']['sec_mean']:.2f} s λ {q['newton']['lam']:.3e}" if 'newton' in q else ""))
        ok &= dec
        if k == "ip":
            ok &= abs(q["E_ip"] - q["obj_last"]) <= 1e-6 * abs(q["obj_last"])
    ok &= len(r) == 2
for q_ip, q_nt in zip(R_["ip"], R_["nt"]):
    out.append(f"[뉴턴 vs 증분 포텐셜] 프레임 {q_ip['t'] + 1}: E/E* 증분 포텐셜 {q_ip['ratio_E_Estar']:.3f} | 뉴턴 목표 L2 {q_nt['ratio_E_Estar']:.3f}"
               f"  ({'뉴턴이 더 내려감' if q_nt['ratio_E_Estar'] < q_ip['ratio_E_Estar'] else '뉴턴이 더 내려가지 않음'})")
print("\n".join(out)); print("TESTOK" if ok else "TESTBAD")
PY
)
  echo "$RES" | tee $U/test_result.txt; echo "[Traceback] $TB" | tee -a $U/test_result.txt
  grep -a "이어 돈다" $TD/rs/B3.log | head -1 | tee -a $U/test_result.txt
  rclone copy $U $R -q
  { grep -q TESTOK <<<"$RES" && [ "$TB" = 0 ] && grep -a -q "이어 돈다" $TD/rs/B3.log; } || fail "시험 실패 (test_result.txt)"
  touch $U/test_$C.ok; echo PF3OBJ_TESTED > $U/PF3OBJ_TESTED; rclone copy $U $R -q; echo PF3OBJ_TESTED; exit 0
fi

# ================= full =================
ls $U/test_*.ok >/dev/null 2>&1 || fail "시험 표식 없음 -- PF3_STAGE=test 먼저"
fitrun(){ K=$1; T=$2; g=$3; O=$U/$K/$T; mkdir -p $O
  [ -f $O/roll_net.pt ] && grep -a -q OVNET_FIT_DONE $O/run_all.log 2>/dev/null && return 0
  CMD=$(fitcmd $T $K $O $g AF_OVNET_R2=$R/$K/$T)
  echo "[실행 명령 $(date '+%F %T') 커밋 $C] $CMD" | tee -a $O/cmd.txt >> $U/PF3OBJ_LAUNCHED
  $CMD >> $O/run_all.log 2>&1 < /dev/null
  grep -a -q OVNET_FIT_DONE $O/run_all.log || { echo "[오류] $K $T 끝나지 않음"; return 1; }
  rclone copyto $O/roll_net.pt $R/$K/$T/roll_net.pt -q; }
render(){ K=$1; T=$2; g=$3; O=$U/$K/$T
  [ -f $O/h5_net/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_net.pt --traj $(gt_of $T) --out $O/h5_net > $O/h5_net.log 2>&1
  [ -f $O/net.mp4 ] || CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $O/h5_net --config $W/lbdata/config.json --shape lego --out $O/net.mp4 > $O/r_net.log 2>&1 < /dev/null
  [ -f $O/net.mp4 ] || { echo "[오류] $K $T 렌더"; return 1; }
  touch $O/.done_render; up; }
psi(){ T=$1; g=$2; D=$U/psi/$T; mkdir -p $D
  [ -f $D/.done ] && return 0
  ( cd $WP && { [ -f $D/col_l2gt.npy ] || CUDA_VISIBLE_DEVICES=$g python -u exe/psi_color_lb.py --gt_dir $(gt_of $T) --pt $(pt_of $T) --dump sp=$S/roll_ov_$T.pt \
        --dump ip=$U/ip/$T/roll_net.pt --dump nt=$U/nt/$T/roll_net.pt --dump l2ov=$U/l2ov/$T/roll_net.pt --dump l2gt=$U/l2gt/$T/roll_net.pt --out $D \
        --eps 1e-3 --q 1 99 --cmap turbo --range $RANGE --stats_name defstats_$T.txt --stats_frames 1 40 80 > $D/psi_color.log 2>&1 < /dev/null; }
    grep -q PSI_COLOR_DONE $D/psi_color.log || { tail -5 $D/psi_color.log; exit 1; }
    for K in ip nt l2ov l2gt; do
      [ -f $D/h5_$K/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $U/$K/$T/roll_net.pt --traj $(gt_of $T) --out $D/h5_$K > $D/h5_$K.log 2>&1
      [ -f $D/psi_${K}_lego.mp4 ] || CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $D/h5_$K --config $W/lbdata/config.json --shape lego \
         --colors $D/col_$K.npy --out $D/psi_${K}_lego.mp4 > $D/r_$K.log 2>&1 < /dev/null
      [ -f $D/psi_${K}_lego.mp4 ] || exit 1; done ) || { echo "[오류] $T psi"; return 1; }
  for n in gt sp; do cp $NS/$T/psi_${n}_lego.mp4 $D/psi_${n}_lego.mp4; done; cp $NS/$T/range.json $D/range_spstick.json
  grep -a "\[색 범위\]" $D/psi_color.log; cat $D/defstats_$T.txt
  touch $D/.done; up; }
job(){ K=$1; T=$2; g=$3; echo "[시작] $K $T GPU$g $(date +%T)"; fitrun $K $T $g && render $K $T $g && touch $U/.done_${K}_$T; echo "[끝] $K $T GPU$g $(date +%T)"; }
declare -A PIDS GPUOF; PEND=""
for K in ip nt l2ov l2gt; do for T in L000 0004; do [ -f $U/.done_${K}_$T ] || PEND="$PEND ${K}_$T"; done; done
busy(){ local b=""; for k in "${!PIDS[@]}"; do kill -0 ${PIDS[$k]} 2>/dev/null && b="$b ${GPUOF[$k]}"; done; echo $b; }
launch_pending(){ # 빈 GPU 가 있는 만큼만 띄우고 나머지는 다음 틱으로 (돌고 있는 실행의 GPU 는 뺀다)
  local np="" kt g K T; for kt in $PEND; do
    g=$(free_gpu $(busy)); [ -z "$g" ] && { np="$np $kt"; continue; }
    K=${kt%_*}; T=${kt#*_}
    ( job $K $T $g >> $U/task_$kt.log 2>&1 ) & PIDS[$kt]=$!; GPUOF[$kt]=$g
    echo "[배정] $K $T GPU$g $(date '+%F %T') $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" | tee -a $U/gpu.log; sleep 30
  done; PEND=$(echo $np); }
launch_pending; up
[ -n "$PEND" ] && echo "[대기] 빈 GPU 가 날 때까지: $PEND $(date '+%F %T')" | tee -a $U/gpu.log
PSI_L=""; PSI_4=""
while :; do
  sleep 300
  [ -n "$PEND" ] && { launch_pending; up; }
  for T in L000 0004; do
    v=PSI_$( [ $T = L000 ] && echo L || echo 4 )
    if [ -z "${!v}" ] && [ ! -f $U/psi/$T/.done ] && [ -f $U/.done_ip_$T ] && [ -f $U/.done_nt_$T ] && [ -f $U/.done_l2ov_$T ] && [ -f $U/.done_l2gt_$T ]; then
      g=$(free_gpu $(busy)); [ -z "$g" ] && continue
      ( psi $T $g >> $U/task_psi_$T.log 2>&1 ) & PIDS[psi_$T]=$!; GPUOF[psi_$T]=$g; printf -v $v '%s' 1
      echo "[배정] psi $T GPU$g $(date '+%F %T')" | tee -a $U/gpu.log; fi
  done
  [ "$(alive "${PIDS[@]}")" = 1 ] && continue
  [ -n "$PEND" ] && continue
  ok=1; for K in ip nt l2ov l2gt; do for T in L000 0004; do [ -f $U/.done_${K}_$T ] || ok=0; done; done
  need=0; [ ! -f $U/psi/L000/.done ] && [ -z "$PSI_L" ] && need=1; [ ! -f $U/psi/0004/.done ] && [ -z "$PSI_4" ] && need=1
  [ $ok = 1 ] && [ $need = 1 ] && continue
  break
done
for K in ip nt l2ov l2gt; do for T in L000 0004; do [ -f $U/.done_${K}_$T ] || fail "$K $T 실패 (task_${K}_$T.log)"; done; done
for T in L000 0004; do [ -f $U/psi/$T/.done ] || fail "psi $T 실패"; done
python - $U > $U/summary.txt 2>&1 <<'PY'
import sys, json, numpy as np
U = sys.argv[1]; S = {}
for k in ("ip", "nt", "l2ov", "l2gt"):
    for T in ("L000", "0004"):
        r = json.load(open(f"{U}/{k}/{T}/records.json"))
        rt = [q["ratio_E_Estar"] for q in r if q.get("ratio_E_Estar") is not None]
        S[f"{k}_{T}"] = dict(pos_err_mean=100 * float(np.mean([q["pos_err_model"] for q in r])), frames=len(r),
                             ratio_E_Estar_median=float(np.median(rt)), ratio_E_Estar_mean=float(np.mean(rt)),
                             ref_err_mean=100 * float(np.mean([q["ref_err"] for q in r])),
                             dist_model_ref_mean=100 * float(np.mean([q["dist_model_ref"] for q in r])),
                             sec_mean=float(np.mean([q["sec"] for q in r])))
        print(f"[{k} {T}] 위치 오차 평균 {S[f'{k}_{T}']['pos_err_mean']:.2f}%  E/E* 중앙 {S[f'{k}_{T}']['ratio_E_Estar_median']:.3f} 평균 {S[f'{k}_{T}']['ratio_E_Estar_mean']:.3f}"
              f"  모델-출력만최적화 거리 평균 {S[f'{k}_{T}']['dist_model_ref_mean']:.2f}%  (출력만 최적화 자체 {S[f'{k}_{T}']['ref_err_mean']:.2f}%)  프레임당 {S[f'{k}_{T}']['sec_mean']:.0f} s")
json.dump(S, open(f"{U}/summary.json", "w"), indent=1)
PY
cat $U/summary.txt
kill $SYNC 2>/dev/null
rclone copy $U $R -q && echo "[R2 업로드 완료] $(date)" && touch $U/ALL_DONE && echo done > $U/PF3OBJ_CLUSTER_DONE && rclone copy $U $R -q
rclone check $U $R --one-way --size-only -q && echo "[R2 대조] 전부 올라감"
echo PF3OBJ_CLUSTER_DONE
