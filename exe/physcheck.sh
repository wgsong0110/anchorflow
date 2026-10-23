#!/bin/bash
# ipL000 물리 점검 (사용자 10-23): lego L000 "한 프레임씩 증분 포텐셜 최적화"(lbours/pf3obj/ip/L000, 커밋 1f22538, --net_inertial 있음) 결과가
#   물리적으로 옳은지 + 마지막 상태에서 출력만 최적화를 이어 돌려 정말 멈추는지.
#   PC   : train_deform AF_PHYSCHECK (단계 step·static) -- pf3obj ip L000 명령줄(cmd.txt) 그대로 (--out·--tag 만 다름; 대조해서 다르면 멈춤)
#   CONT : train_deform AF_OVCONT_FIT -- spstick.sh BASE + OV (출력만 최적화 표준, spstick roll_ov 와 같은 명령줄) 로 roll_net.pt 마지막 상태에서 N 프레임
#          (프레임 파일로 이어 돈다; 먼저 짧은 이어 돌기 시험: 한 번에 2 프레임 vs 1 -> 다시 띄워 2 프레임)
#   끝   : PC series (+CONT) -> physcheck_report.py 그림·표 -> lb_roll_to_h5 --first_from_dump + lb_render (프레임 번호 빨간색)
# 멱등: 다시 띄우면 끝난 단계는 건너뛰고 각 실행은 이어 돈다. 출력은 R2 lbours/ipL000_physcheck 와 통째 동기화 (제외 없음, 5 분).
# 잡 안에서만: AF_DETACH=1 bash exe/job_run.sh $W/physcheck_boot.sh <로그>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
U=$W/lbours/ipL000_physcheck; IP=$W/lbours/pf3obj/ip/L000; S=$W/lbours/spstick
R=r2:storage/result/anchorflow/lbours/ipL000_physcheck; mkdir -p $U
NC=${PHYSCHECK_CONT_N:-120}
if [ -f $U/driver.hb ] && [ $(( $(date +%s) - $(cat $U/driver.hb) )) -lt 300 ]; then echo "[중단] 다른 드라이버가 돌고 있다 $(date)"; exit 0; fi
date +%s > $U/driver.hb
( while true; do date +%s > $U/driver.hb; sleep 120; done ) & HB=$!
( while true; do sleep 300; rclone copy $U $R -q; done ) & SYNC=$!
trap 'kill $SYNC $HB 2>/dev/null' EXIT
echo "[시작] $(date) 커밋 $C 노드 $(hostname) 잡 ${PBS_JOBID:-?}"
fail(){ echo "[오류] $1 $(date)"; echo "$1 ($(date '+%F %T'), 커밋 $C)" > $U/PHYSCHECK_FAIL; rclone copy $U $R -q; exit 1; }
# ---- 명령줄 ----
BASE=$(P=$W/lbpool bash -c "$(grep -E '^(COMMON|REST_OLD|FLOOR_SP|REST|BASE_SP|BASE_NOXN|BASE)=' $W/spstick.sh; echo 'echo "$BASE"')")
OV=$(bash -c "$(grep -E '^OV=' $W/spstick.sh); echo \"\$OV\"")
[ "$OV" = "--out_var --ov_init predict --ov_roll 500 --ov_opt lbfgs --ov_vars out --ov_obj ip" ] || fail "spstick OV 를 못 읽었다"
[ "[BASE] $BASE" = "$(grep -a -m1 '^\[BASE\]' $W/spstick.out)" ] || fail "BASE 가 spstick.out 과 다르다"
XA="--out_var --ov_init predict --ov_vars net --ov_opt adam --out_var_lr 1e-3 --ov_roll 2000 --ov_obj ip"
PCARGS="--infer --data $W/one/L000 --hold_traj L000 --out $U/pc_run --tag PC $BASE $XA --iters 0 --eval_t0 0 --eval_len 80 --tb none"
ORIG=$(sed -n 's/.*python -u exe\/train_deform.py //p' $IP/cmd.txt | head -1)
tokdiff(){ comm -$1 <(tr ' ' '\n' <<<"$2" | sort) <(tr ' ' '\n' <<<"$3" | sort) | paste -sd' '; }
DM_=$(tokdiff 23 "$ORIG" "$PCARGS"); DP_=$(tokdiff 13 "$ORIG" "$PCARGS")
echo "[PC 명령줄 vs pf3obj ip L000 cmd.txt] 빠짐: '$DM_' | 더함: '$DP_'" | tee $U/cmdline.txt
[ "$DM_" = "$(tr ' ' '\n' <<<"$IP/run OVN" | sort | paste -sd' ')" ] && [ "$DP_" = "$(tr ' ' '\n' <<<"$U/pc_run PC" | sort | paste -sd' ')" ] || fail "PC 명령줄이 pf3obj ip 와 --out/--tag 말고도 다르다"
cargs(){ echo "--infer --data $W/one/L000 --hold_traj L000 --out $1 --tag OVC $BASE $OV --iters 0 --eval_t0 0 --eval_len 80 --tb none"; }
echo "[CONT 명령줄 vs spstick ov] spstick.sh: 'python -u exe/train_deform.py --infer --data \$W/one/\$T --hold_traj \$T --out \$U/ov_\$T --tag OVN \$BASE \$OV ...' 같은 BASE·OV, --out·--tag 만 다름" | tee -a $U/cmdline.txt
PCENV="AF_PHYSCHECK=$U AF_PHYSCHECK_FRAMES=$IP/frames AF_PHYSCHECK_MODEL=$IP/roll_net.pt AF_PHYSCHECK_OV=$S/roll_ov_L000.pt AF_PHYSCHECK_GTDIR=$W/lbdata/test_low/L000 AF_PHYSCHECK_ESTAR=500"
echo "[PC] env $PCENV python -u exe/train_deform.py $PCARGS" | tee -a $U/cmdline.txt
echo "[CONT] env AF_OVCONT_FIT=$U/cont AF_OVCONT_FROM=$IP/roll_net.pt AF_OVCONT_N=$NC python -u exe/train_deform.py $(cargs $U/cont_run)" | tee -a $U/cmdline.txt
lowgpu(){ # 메모리 가장 적은 카드 (빈 카드 없음 -- 사용자 10-23: 속도 측정 아님, 경합 괜찮음); 인자 카드는 되도록 피한다
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | awk -F', *' -v ex=" $* " 'index(ex," "$1" ")==0{print $1; exit}'; }
g1=$(lowgpu); g2=$(lowgpu $g1); [ -n "$g1" ] && [ -n "$g2" ] || fail "GPU 를 못 골랐다"
echo "[GPU] PC $g1, CONT $g2  $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" | tee -a $U/gpu.log
# ---- PC (step, static) ----
( [ -f $U/.pc_done ] || { env $PCENV AF_PHYSCHECK_STAGES=step,static CUDA_VISIBLE_DEVICES=$g1 python -u exe/train_deform.py $PCARGS >> $U/pc.log 2>&1 < /dev/null
    grep -a -q PHYSCHECK_DONE $U/pc.log && touch $U/.pc_done; } ) & PPC=$!
# ---- CONT: 이어 돌기 시험 -> 본 실행 ----
(
  T1=$U/test_resume
  if [ ! -f $T1/.ok ]; then
    mkdir -p $T1/A $T1/B
    [ -f $T1/A/frames/cont_001.pt ] || env AF_OVCONT_FIT=$T1/A AF_OVCONT_FROM=$IP/roll_net.pt AF_OVCONT_N=2 CUDA_VISIBLE_DEVICES=$g2 python -u exe/train_deform.py $(cargs $T1/A/run) > $T1/A.log 2>&1 < /dev/null
    [ -f $T1/B/frames/cont_000.pt ] || env AF_OVCONT_FIT=$T1/B AF_OVCONT_FROM=$IP/roll_net.pt AF_OVCONT_N=1 CUDA_VISIBLE_DEVICES=$g2 python -u exe/train_deform.py $(cargs $T1/B/run) > $T1/B1.log 2>&1 < /dev/null
    [ -f $T1/B/frames/cont_001.pt ] || env AF_OVCONT_FIT=$T1/B AF_OVCONT_FROM=$IP/roll_net.pt AF_OVCONT_N=2 CUDA_VISIBLE_DEVICES=$g2 python -u exe/train_deform.py $(cargs $T1/B/run) > $T1/B2.log 2>&1 < /dev/null
    python - $T1 $IP/roll_net.pt <<'PY' > $T1/result.txt 2>&1
import sys, torch
t1, rn = sys.argv[1:]
L = lambda p: torch.load(p, map_location="cpu", weights_only=False)
a1, b1 = L(f"{t1}/A/frames/cont_001.pt"), L(f"{t1}/B/frames/cont_001.pt")
a0 = L(f"{t1}/A/frames/cont_000.pt"); D = torch.load(rn, map_location="cpu", weights_only=False, mmap=True)
ext = float(a1["EXT"])
dx = float((a1["x"] - b1["x"]).norm(dim=-1).max()) / ext
mv = float((a1["x"] - D["pred"][-1].float()).norm(dim=-1).mean()) / ext
print(f"[이어 돌기] 프레임 {a1['frame']}: 한 번에 vs 끊어서 위치 최대차 {100 * dx:.3e}% EXT (2 스텝 동안 평균 이동 {100 * mv:.3e}%), F 최대차 {float((a1['F'] - b1['F']).abs().max()):.3e}")
print(f"[시작 상태] 프레임 {a0['frame'] - 1} = roll_net 마지막 (pred {tuple(D['pred'].shape)})")
ok = dx <= max(1e-3 * mv, 1e-7)
print("RESUMEOK" if ok else "RESUMEBAD")
PY
    cat $T1/result.txt
    # 이어 돈 경로를 실제로 탔으면 진행 (위치차는 기록만 -- GPU 원자 덧셈 비결정성 크기는 보고에서 판단)
    grep -q -a "이어 돈다" $T1/B2.log && grep -q -a OVCONT_FIT_DONE $T1/B2.log && touch $T1/.ok
  fi
  [ -f $T1/.ok ] || { echo "[오류] 이어 돌기 시험 실패 ($T1/result.txt)"; exit 1; }
  [ -f $U/cont/roll_cont.pt ] || env AF_OVCONT_FIT=$U/cont AF_OVCONT_FROM=$IP/roll_net.pt AF_OVCONT_N=$NC CUDA_VISIBLE_DEVICES=$g2 python -u exe/train_deform.py $(cargs $U/cont_run) >> $U/cont.log 2>&1 < /dev/null
  grep -a -q OVCONT_FIT_DONE $U/cont.log && touch $U/.cont_done
) > $U/cont_driver.log 2>&1 & PCT=$!
wait $PPC; wait $PCT
cat $U/cont_driver.log
[ -f $U/.pc_done ] || fail "PC (step/static) 실패 -- pc.log"
[ -f $U/.cont_done ] || fail "CONT 실패 -- cont.log / cont_driver.log"
rclone copy $U $R -q
# ---- series + 그림·표 + 렌더 ----
[ -f $U/series.json ] && grep -q '"cont"' $U/series.json || env $PCENV AF_PHYSCHECK_STAGES=series AF_PHYSCHECK_CONT=$U/cont CUDA_VISIBLE_DEVICES=$g1 \
   python -u exe/train_deform.py $PCARGS >> $U/pc_series.log 2>&1 < /dev/null
grep -a -q PHYSCHECK_DONE $U/pc_series.log || fail "series 실패"
python exe/physcheck_report.py --dir $U --cont $U/cont --png $U/ipL000_physcheck.png --md $U/ipL000_physcheck.md > $U/report.log 2>&1 || fail "그림·표 실패"
[ -f $U/h5_cont/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $U/cont/roll_cont.pt --traj $W/lbdata/test_low/L000 --out $U/h5_cont --first_from_dump > $U/h5_cont.log 2>&1
[ -f $U/ipL000_rest_ovcont.mp4 ] || CUDA_VISIBLE_DEVICES=$g1 python -u exe/lb_render.py --sim $U/h5_cont --config $W/lbdata/config.json --shape lego \
   --out $U/ipL000_rest_ovcont.mp4 --frame_label 80 --caption "한 프레임씩 결과 마지막 상태(80) → 출력만 최적화 이어 돌리기" > $U/render.log 2>&1 < /dev/null
[ -f $U/ipL000_rest_ovcont.mp4 ] || fail "렌더 실패"
kill $SYNC 2>/dev/null
rclone copy $U $R -q && touch $U/PHYSCHECK_DONE && rclone copy $U $R -q
rclone check $U $R --one-way --size-only -q && echo "[R2 대조] 전부 올라감"
echo PHYSCHECK_CLUSTER_DONE
