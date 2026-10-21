#!/bin/bash
# 관성 예측 없이 시작하는 출력만 최적화 (사용자 10-21): lego 새 바닥(softplus + 미끄러짐 방지) spstick.sh 의 ov 단계와
# **글자 그대로 같은 명령**에서 --ov_init 만 바꾼다.
#   z: --ov_init zero  (매 프레임 노드 변위 0 = 변형 없음에서 출발)
#   n: --ov_init randn --ov_init_std 2.8284e-05  (평균 0 가우시안, 표준편차 = 1e-3 × node_h 0.028284; 시드 = --seed 기본 0 고정)
#   기존 p: --ov_init predict (노드 관성 예측 τ·v_n + τ²g) = lbours/spstick/roll_ov_{L000,0004}.pt (다시 돌리지 않음)
# 출력만 최적화 경로에서 --net_inertial 은 매개화에 들어가지 않는다: 자유변수가 망을 건너뛰는 dp_fast 로 들어가
# (deform_model.advance 의 `if out is None:` 블록 밖) 관성 예측은 오직 --ov_init predict 의 시작점으로만 쓰인다. 그래서 --net_inertial 은 그대로 둔다.
# 코드 = 13471f4 (spstick 과 같은 커밋) + 진단 기록 AF_OV_STOP_LOG 만 (계산 불변).
# 작업: (1) 롤아웃 z·n × L000·0004 = 4 실행 (빈 카드에 나눠 동시에), 덤프 F32
#       (2) 같은 상태 비교: 기존 p 롤아웃 덤프의 프레임 시작 상태 80 개에서 p·z·n 세 시작점으로 같은 L-BFGS 500 (AF_OV_MIN_STATES)
#       (3) 3DGS 렌더(z·n), ψ 색(공통 고정 범위 lbours/lego_psi/range.json) 정답·기존·z·n 렌더, 집계 summary.txt
# 출력은 처음부터 R2 lbours/ov_noinertial 과 통째 동기화 (제외 없음, 5분), 큰 덤프는 생기는 즉시 백그라운드 copyto.
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
C=${OVNOIN_C:?커밋}; WT=$W/wt/ovnoin_$C; WS=$W/wt/spstick_13471f4; WP=$W/wt/psi_d993fb5
[ -d $WT ] || (cd $W/anchorflow && git fetch -q origin && git worktree add -q --detach $WT $C) || { echo "worktree 실패"; exit 1; }
P=$W/lbpool; S=$W/lbours/spstick; U=$W/lbours/ov_noinertial; R=r2:storage/result/anchorflow/lbours/ov_noinertial
RANGE=$W/lbours/lego_psi/range.json; STD=2.8284e-05
mkdir -p $U
fail(){ echo "[실패] $1 $(date)"; echo "$1 $(date)" > $U/OVNOIN_FAIL; rclone copy $U $R -q; exit 1; }
cd $WT || exit 1
echo "[시작] $(date) 커밋 $(git rev-parse --short HEAD) (13471f4 대비: $(git diff --stat 13471f4 HEAD | tail -1))"
git diff 13471f4 HEAD -- exe/train_deform.py > $U/code_diff_vs_13471f4.patch
# ---- spstick.sh 의 정의를 **원문에서 그대로** 가져온다 (P 는 같은 $W/lbpool) ----
DEF=$(grep -E '^(COMMON|REST_OLD|FLOOR_SP|REST|BASE_SP|BASE_NOXN|BASE|OV)=' $W/spstick.sh)
[ $(echo "$DEF" | wc -l) = 8 ] || fail "spstick.sh 정의 8 줄을 못 찾음"
eval "$DEF"
echo "$DEF" > $U/spstick_defs.txt
OVZ="${OV/--ov_init predict/--ov_init zero}"
OVN="${OV/--ov_init predict/--ov_init randn --ov_init_std $STD}"
[ "$OVZ" != "$OV" ] && [ "$OVN" != "$OV" ] || fail "OV 치환 실패"
ovf(){ case $1 in z) echo "$OVZ";; n) echo "$OVN";; p) echo "$OV";; esac; }
gt_of(){ [ $1 = L000 ] && echo $W/lbdata/test_low/L000 || echo $W/lbdata/test/$1; }
pt_of(){ [ $1 = L000 ] && echo $W/lbpt/test_low/L000.pt || echo $W/lbpt/test/$1.pt; }
ORIG(){ echo "python -u exe/train_deform.py --infer --data $W/one/$1 --hold_traj $1 --out $S/ov_$1 --tag OVN $BASE $OV --iters 0 --eval_t0 0 --eval_len 80 --tb none"; }
NEWC(){ echo "python -u exe/train_deform.py --infer --data $W/one/$2 --hold_traj $2 --out $U/ov_$1_$2 --tag OVN $BASE $(ovf $1) --iters 0 --eval_t0 0 --eval_len 80 --tb none"; }
MSC(){ echo "python -u exe/train_deform.py --infer --data $W/one/$2 --hold_traj $2 --out $U/msrun_$1_$2 --tag OVN $BASE $(ovf $1) --iters 0 --eval_t0 0 --eval_len 80 --tb none"; }
# ---- 명령줄 자동 대조 (토큰 위치별 diff) ----
{
  echo "# spstick.sh ov 원래 명령 vs 이번 (같은 위치 토큰 diff; '<' 원래, '>' 이번)"
  for T in L000 0004; do for k in z n; do
    echo "== $k $T"; diff <(ORIG $T | tr ' ' '\n') <(NEWC $k $T | tr ' ' '\n') | grep '^[<>]'
  done; for k in p z n; do echo "== 같은 상태 $k $T"; diff <(ORIG $T | tr ' ' '\n') <(MSC $k $T | tr ' ' '\n') | grep '^[<>]'; done; done
  echo "# 환경변수: 원래 AF_INV_LOG AF_TUNNEL_LOG AF_ROLL_DUMP AF_ROLL_DUMP_F=32 AF_ROLL_TAG AF_ROLL_T0=0 + 이번 AF_OV_STOP_LOG (진단 기록만)"
  echo "# 같은 상태 실행은 AF_ROLL_DUMP 대신 AF_OV_MIN_STATES·AF_IP_STATES=$S/roll_ov_<T>.pt"
  echo "# 코드: 13471f4 대비 train_deform.py 변경 = AF_OV_STOP_LOG 기록만 ($(grep -c '^[+-][^+-]' $U/code_diff_vs_13471f4.patch) 줄)"
} > $U/cmdline_diff.txt
cat $U/cmdline_diff.txt
# 허용된 차이만 있는지 검사: --out 경로, --ov_init 값, (n) --ov_init_std 값
BAD=$(grep '^[<>]' $U/cmdline_diff.txt | grep -v -E "^[<>] ($S/ov_|$U/ov_|$U/msrun_|predict$|zero$|randn$|--ov_init_std$|$STD$|--ov_roll$)" | head -3)
# randn 은 토큰 2 개(--ov_init_std 값)가 더 들어가 위치가 밀린다 -> 밀린 토큰 --ov_roll 등은 원래와 같은지 집합으로 다시 확인
for T in L000 0004; do
  X=$(comm -3 <(ORIG $T | tr ' ' '\n' | sort) <(NEWC n $T | tr ' ' '\n' | sort) | tr -d '\t' | sort | paste -sd' ')
  echo "[n $T 집합 차이] $X" | tee -a $U/cmdline_diff.txt
done
[ -z "$BAD" ] || echo "[참고] 위치 diff 의 기타 줄: $BAD" | tee -a $U/cmdline_diff.txt
for T in L000 0004; do
  X=$(comm -3 <(ORIG $T | tr ' ' '\n' | sort) <(NEWC z $T | tr ' ' '\n' | sort) | tr -d '\t' | sort | paste -sd' ')
  [ "$X" = "$(printf '%s\n' $S/ov_$T $U/ov_z_$T predict zero | sort | paste -sd' ')" ] || fail "z $T 명령 차이가 예상과 다름: $X"
  X=$(comm -3 <(ORIG $T | tr ' ' '\n' | sort) <(NEWC n $T | tr ' ' '\n' | sort) | tr -d '\t' | sort | paste -sd' ')
  [ "$X" = "$(printf '%s\n' $S/ov_$T $U/ov_n_$T predict randn --ov_init_std $STD | sort | paste -sd' ')" ] || fail "n $T 명령 차이가 예상과 다름: $X"
done
echo "[대조 통과] 차이 = --out, --ov_init (zero / randn --ov_init_std $STD) 뿐" | tee -a $U/cmdline_diff.txt
( while true; do rclone copy $U $R -q; sleep 300; done ) & SYNC=$!
up(){ rclone copy $U $R -q --include "*.mp4" --include "*.txt" --include "*.json" --include "*.jsonl" --include "OVNOIN_*"; }
bgup(){ ( rclone copyto $1 $R/${1#$U/} -q && echo "[R2 덤프 올림] ${1#$U/} $(date +%T)" ) & }
# 실제로 뜬 프로세스의 argv 를 의도한 명령과 대조
argv_ck(){ pid=$1; want=$2; f=$3; sleep 20; [ -r /proc/$pid/cmdline ] || { echo "[argv] pid $pid 이미 끝남" > $f; return; }
  got=$(tr '\0' ' ' < /proc/$pid/cmdline | sed 's/ $//; s/^[^ ]*python[^ ]* /python /' | tr -s ' '); want=$(echo "$want" | tr -s ' ' | sed 's/ $//')
  [ "$got" = "$want" ] && echo "[argv 일치] $got" > $f || { echo "[argv 불일치]"; echo "want: $want"; echo "got:  $got"; } > $f; }
# ---- 짧은 시험 (반복 3, 2 프레임 / 같은 상태 80 개): 기록이 남는지, 시작점 크기가 0 / ~1e-4 / 관성 예측인지 ----
smoke(){ g=$1; ST=$U/smoke; rm -rf $ST; mkdir -p $ST; T=L000
  for k in p z n; do
    env AF_OV_STOP_LOG=$ST/stop_$k.jsonl AF_ROLL_TAG=$T AF_ROLL_T0=0 CUDA_VISIBLE_DEVICES=$g python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $ST/ov_$k --tag OVN $BASE \
      $(ovf $k | sed 's/--ov_roll 500/--ov_roll 3/') --iters 0 --eval_t0 0 --eval_len 2 --tb none > $ST/r_$k.log 2>&1 < /dev/null
    echo "[시험 $k] $(grep -a '출력변수' $ST/r_$k.log | cut -c1-90) | 기록 $(wc -l < $ST/stop_$k.jsonl 2>/dev/null) 줄 | $(head -1 $ST/stop_$k.jsonl 2>/dev/null | cut -c1-260) | Traceback $(grep -a -c Traceback $ST/r_$k.log)"
  done
  env AF_OV_STOP_LOG=$ST/stop_ms.jsonl AF_OV_MIN_STATES=$ST/ms.json AF_IP_STATES=$S/roll_ov_$T.pt CUDA_VISIBLE_DEVICES=$g python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $ST/ms --tag OVN $BASE \
      $(ovf z | sed 's/--ov_roll 500/--ov_roll 3/') --iters 0 --eval_t0 0 --eval_len 80 --tb none > $ST/r_ms.log 2>&1 < /dev/null
  echo "[시험 같은 상태] 상태 $(python -c "import json;print(len(json.load(open('$ST/ms.json'))))" 2>/dev/null) 개, 기록 $(wc -l < $ST/stop_ms.jsonl 2>/dev/null) 줄, Traceback $(grep -a -c Traceback $ST/r_ms.log)"
  python - $ST <<'PY'
import json, sys
d = sys.argv[1]
r = {k: [json.loads(l) for l in open(f"{d}/stop_{k}.jsonl")] for k in ("p", "z", "n")}
ms = json.load(open(f"{d}/ms.json"))
ok = all(len(r[k]) == 2 for k in r) and len(ms) == 80
ok = ok and r["z"][0]["init_absmax"] == 0.0 and 0 < r["n"][0]["init_absmax"] < 1e-3 and r["p"][0]["init_absmax"] > 1e-6 and r["p"][0]["init_rms"] != r["n"][0]["init_rms"]
print("[시험 판정]", "통과" if ok else "실패", {k: (r[k][0]["init_absmax"], r[k][0]["init_rms"], r[k][0]["why"], r[k][0]["iters"]) for k in r})
sys.exit(0 if ok else 1)
PY
}
roll(){ k=$1; T=$2; g=$3; N=${k}_$T
  if [ ! -f $U/roll_ov_$N.pt ]; then rm -f $U/inv_ov_$N.jsonl $U/tun_ov_$N.jsonl $U/stop_$N.jsonl
    echo "[롤아웃 시작] $k $T GPU$g $(date +%T)"
    env AF_INV_LOG=$U/inv_ov_$N.jsonl AF_TUNNEL_LOG=$U/tun_ov_$N.jsonl AF_OV_STOP_LOG=$U/stop_$N.jsonl AF_ROLL_DUMP=$U/roll_ov_$N.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 \
      CUDA_VISIBLE_DEVICES=$g $(NEWC $k $T) > $U/v_$N.log 2>&1 < /dev/null & pp=$!
    argv_ck $pp "$(NEWC $k $T)" $U/argv_$N.txt; cat $U/argv_$N.txt; wait $pp
    [ -f $U/roll_ov_$N.pt ] && bgup $U/roll_ov_$N.pt; fi
  grep -a "롤아웃\] $T\|Traceback\|Error" $U/v_$N.log | tail -2 | cut -c1-200
  echo "$N $(grep -a "롤아웃\] $T" $U/v_$N.log | tail -1 | grep -o '평균 [0-9.]*%')" > $U/err_$N.txt; up
  [ -f $U/roll_ov_$N.pt ] || return 1
  [ -f $U/h5_ov_$N/meta.json ] || ( cd $WS && python -u exe/lb_roll_to_h5.py --dump $U/roll_ov_$N.pt --traj $(gt_of $T) --out $U/h5_ov_$N > $U/h5_ov_$N.log 2>&1 < /dev/null )
  [ -f $U/ov_$N.mp4 ] || ( cd $WS && CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $U/h5_ov_$N --config $W/lbdata/config.json --shape lego --out $U/ov_$N.mp4 > $U/r_ov_$N.log 2>&1 < /dev/null )
  ls -la $U/ov_$N.mp4; up; }
ms(){ k=$1; T=$2; g=$3; N=${k}_$T
  [ -f $U/ms_$N.json ] && [ $(python -c "import json;print(len(json.load(open('$U/ms_$N.json'))))") = 80 ] && return 0
  rm -f $U/stop_ms_$N.jsonl; echo "[같은 상태 시작] $k $T GPU$g $(date +%T)"
  env AF_OV_STOP_LOG=$U/stop_ms_$N.jsonl AF_OV_MIN_STATES=$U/ms_$N.json AF_IP_STATES=$S/roll_ov_$T.pt CUDA_VISIBLE_DEVICES=$g $(MSC $k $T) > $U/v_ms_$N.log 2>&1 < /dev/null & pp=$!
  argv_ck $pp "$(MSC $k $T)" $U/argv_ms_$N.txt; cat $U/argv_ms_$N.txt; wait $pp
  grep -a "상태 최솟값 기록\|Traceback\|Error" $U/v_ms_$N.log | tail -2; up; }
psi(){ T=$1; g=$2; D=$U/psi/$T; mkdir -p $D; cp $RANGE $D/range_fixed_from_lego_psi.json
  [ -f $D/psi_hist.npz ] || ( cd $WP && CUDA_VISIBLE_DEVICES=$g python -u exe/psi_color_lb.py --gt_dir $(gt_of $T) --pt $(pt_of $T) \
     --dump old=$S/roll_ov_$T.pt --dump z=$U/roll_ov_z_$T.pt --dump n=$U/roll_ov_n_$T.pt --out $D --eps 1e-3 --q 1 99 --cmap turbo \
     --range $RANGE --stats_name defstats_$T.txt --stats_frames 1 40 80 > $D/psi_color.log 2>&1 < /dev/null )
  grep -v "it/s\|s/it" $D/psi_color.log | tail -12
  grep -q PSI_COLOR_DONE $D/psi_color.log || return 1
  for n in gt old z n; do
    case $n in gt) SIM=$(gt_of $T);; old) SIM=$S/h5_ov_$T;; *) SIM=$U/h5_ov_${n}_$T;; esac
    [ -f $D/psi_${n}_lego.mp4 ] || ( cd $WP && CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $SIM --config $W/lbdata/config.json --shape lego \
       --colors $D/col_$n.npy --out $D/psi_${n}_lego.mp4 > $D/r_$n.log 2>&1 < /dev/null )
    tail -1 $D/r_$n.log; done
  ls $D/psi_*_lego.mp4 | wc -l | grep -qx 4 && touch $D/DONE; up; rclone copy $D $R/psi/$T -q; }
task(){ name=$1; g=$2; echo "[시작] $name GPU$g $(date +%T)"
  case $name in
  z_*|n_*) roll ${name%%_*} ${name#*_} $g ;;
  ms_*) x=${name#ms_}; ms ${x%%_*} ${x#*_} $g ;;
  psi_*) psi ${name#psi_} $g ;;
  esac
  echo "[끝] $name GPU$g $(date +%T)"; touch $U/.done_$name; }
freeg(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$1!=1 && $2<1024{print $1}'; }
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee $U/gpu_at_start.txt
G0=$(freeg | head -1); [ -n "$G0" ] || fail "빈 GPU 없음"
smoke $G0 > $U/smoke.log 2>&1; cat $U/smoke.log
grep -q '시험 판정\] 통과' $U/smoke.log || fail "짧은 시험 실패 (smoke.log)"
up
declare -A BUSY
Q=(); for t in z_L000 z_0004 n_L000 n_0004 ms_z_L000 ms_z_0004 ms_n_L000 ms_n_0004 ms_p_L000 ms_p_0004; do [ -f $U/.done_$t ] || Q+=($t); done
declare -A QD; L=0
while :; do
  date +%s > $U/heartbeat; rclone copyto $U/heartbeat $R/heartbeat -q
  for g in "${!BUSY[@]}"; do kill -0 ${BUSY[$g]} 2>/dev/null || unset BUSY[$g]; done
  for T in L000 0004; do
    [ -f $U/.done_z_$T ] && [ -f $U/.done_n_$T ] && [ ! -f $U/.done_psi_$T ] && [ -z "${QD[$T]:-}" ] && { Q=(psi_$T "${Q[@]}"); QD[$T]=1; }; done
  [ ${#Q[@]} -eq 0 ] && [ ${#BUSY[@]} -eq 0 ] && break
  for g in $(freeg); do
    [ ${#Q[@]} -eq 0 ] && break; [ -n "${BUSY[$g]:-}" ] && continue
    task ${Q[0]} $g >> $U/task_${Q[0]}.log 2>&1 & BUSY[$g]=$!; echo "[배정] ${Q[0]} -> GPU$g $(date +%T)"; Q=("${Q[@]:1}")
    [ $L = 0 ] && { L=1; ( sleep 60; echo "커밋 $C, 첫 배정 $(date)" > $U/OVNOIN_LAUNCHED; rclone copy $U/OVNOIN_LAUNCHED $R -q ) & }
    sleep 45
  done
  sleep 120
done
for k in z_L000 z_0004 n_L000 n_0004; do cat $U/err_$k.txt; done
python -u $WT/exe/ov_noinertial_ana.py --dir $U --spstick $S > $U/ana.log 2>&1 < /dev/null; cat $U/ana.log
grep -q OVNOIN_ANA_DONE $U/ana.log || fail "집계 실패 (ana.log)"
for k in z_L000 z_0004 n_L000 n_0004; do [ -f $U/ov_$k.mp4 ] || fail "3DGS 렌더 없음 $k"; done
for T in L000 0004; do [ -f $U/psi/$T/DONE ] || fail "psi 렌더 없음 $T"; done
wait
kill $SYNC 2>/dev/null
rclone copy $U $R -q && echo "[R2 업로드 완료] $(date)"
rclone check $U $R --one-way --size-only -q && echo "[R2 대조 통과]" || fail "R2 대조 실패"
echo "$(date)" > $U/OVNOIN_CLUSTER_DONE; rclone copy $U $R -q
echo OVNOIN_CLUSTER_END
