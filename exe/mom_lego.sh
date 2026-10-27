#!/bin/bash
# 운동량 잔차 변위 망 lego L000 (사용자 10-27 확정 설계, doc/mom_lego.md, exe/train_mom.py).
#   bash exe/mom_lego.sh test <GPU>   생성 분포 vs 실제 롤아웃 분포 (cover) -> 이어 돌리기 시험 (20 -> 30 vs 한 번에 30) -> 짧은 학습 (MOM_SMOKE_IT, 기본 300)
#                                     -> 그 체크포인트 롤아웃·h5·3DGS 렌더 시험. 출력 lbours/mom_lego/test_<커밋>, 표식 MOM_TEST_DONE / MOM_TEST_FAIL
#   bash exe/mom_lego.sh run <GPU>    본 학습 (MOM_ITERS, 기본 30000; 1000 마다 80 프레임 롤아웃 평가 -> TB) -> 최저 오차 체크포인트로 80 프레임 롤아웃 덤프
#                                     -> 3DGS 렌더 (정답 | 변위 망, 프레임 번호·한국어 라벨) -> 기존 결과와 비교 err_cmp.txt.
#                                     출력 lbours/mom_lego/run, 표식 MOM_LEGO_LAUNCHED, MOM_LEGO_DONE (끝), MOM_LEGO_FAIL
#   잡 안에서만: AF_DETACH=1 bash exe/job_run.sh <부트 스크립트> <로그>. 멱등: 다시 띄우면 끝난 단계는 건너뛰고 학습은 ck.pt 에서 이어 돈다.
#   출력은 처음부터 R2 와 통째 동기화 (제외 없음, 3 분 루프; 체크포인트·덤프는 저장 즉시 따로 복사).
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
STAGE=${1:?test 또는 run}; G=${2:?GPU}
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
PT=$W/one/L000/L000.pt; GTD=$W/lbdata/test_low/L000; CFG=$W/lbdata/config.json
FEMD=$W/lbours/fem2_lego/run/roll_fem.pt; GTV=$W/lbours/fem2_lego/gt_lego.mp4
U=$W/lbours/mom_lego; R=r2:storage/result/anchorflow/lbours/mom_lego; mkdir -p $U
case $STAGE in test) O=$U/test_$C; MK=MOM_TEST ;; run) O=$U/run; MK=MOM_LEGO ;; *) echo "[오류] 단계 $STAGE"; exit 1 ;; esac
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
for f in $PT $GTD/meta.json $CFG; do [ -e $f ] || fail "입력 없음 $f"; done
while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 500 ]; do echo "[대기] GPU$G 사용 중 $(date +%T)"; sleep 120; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $O/gpu_at_start.txt
TM="python -u exe/train_mom.py --pt $PT --node_h 0.028284 --commit $C"
ck_it(){ python -c "import torch;print(torch.load('$1',map_location='cpu',weights_only=False)['it'])" 2>/dev/null; }
render(){ # render <체크포인트> <디렉토리> <라벨>
  local CK=$1 D=$2 LB=$3; mkdir -p $D
  if ! grep -q MOM_ROLL_DONE $D/roll.log 2>/dev/null; then
    echo "[롤아웃 명령] CUDA_VISIBLE_DEVICES=$G $TM --mode roll --resume $CK --out $D --r2 $RO/${D#$O/} --eval_len 80" > $D/cmd_roll.txt
    CUDA_VISIBLE_DEVICES=$G $TM --mode roll --resume $CK --out $D --r2 $RO/${D#$O/} --eval_len 80 > $D/roll.log 2>&1 < /dev/null
  fi
  grep -a "\[롤아웃\] L000\|Traceback\|Error" $D/roll.log | tail -2 | cut -c1-300
  grep -q MOM_ROLL_DONE $D/roll.log || return 1
  [ -f $D/h5/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $D/roll_mom.pt --traj $GTD --out $D/h5 > $D/h5.log 2>&1
  [ -f $D/mom_lego.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $D/h5 --config $CFG --shape lego --out $D/mom_lego.mp4 \
     --frame_label 0 --caption "$LB" < /dev/null > $D/r_mom.log 2>&1
  [ -f $D/mom_lego.mp4 ] || { echo "[오류] 렌더 $(tail -3 $D/r_mom.log)"; return 1; }
  [ -f $D/gt_lego.mp4 ] || cp $GTV $D/gt_lego.mp4 2>/dev/null || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $GTD --config $CFG --shape lego \
     --out $D/gt_lego.mp4 --frame_label 0 --caption "정답" < /dev/null > $D/r_gt.log 2>&1
  python - $D/gt_lego.mp4 $D/mom_lego.mp4 $D/cmp_lego.mp4 <<'PY' > $D/cmp.log 2>&1
import sys, numpy as np, imageio.v2 as iio
ra, rb = iio.get_reader(sys.argv[1]), iio.get_reader(sys.argv[2])
fps = ra.get_meta_data().get("fps", 30)
A, Bv = [f for f in ra], [f for f in rb]
n = max(len(A), len(Bv)); w = iio.get_writer(sys.argv[3], fps=fps, codec="libx264", quality=8, macro_block_size=1)
for i in range(n):
    a, b = A[min(i, len(A) - 1)], Bv[min(i, len(Bv) - 1)]
    hh = max(a.shape[0], b.shape[0])
    pad = lambda f: np.pad(f, ((0, hh - f.shape[0]), (0, 0), (0, 0)), constant_values=255)
    w.append_data(np.concatenate([pad(a), pad(b)], 1))
w.close(); print(f"[저장] {sys.argv[3]} 프레임 {n} (정답 {len(A)}, 변위 망 {len(Bv)})")
PY
  cat $D/cmp.log; [ -f $D/cmp_lego.mp4 ]; }

if [ $STAGE = test ]; then
  # 1) 분포 비교 + 노드 힘 검산
  if ! grep -q MOM_COVER_DONE $O/cover/cover.log 2>/dev/null; then mkdir -p $O/cover
    CUDA_VISIBLE_DEVICES=$G $TM --mode cover --out $O/cover --fem_dump $FEMD --n_cover ${MOM_NCOVER:-256} > $O/cover/cover.log 2>&1 < /dev/null; fi
  grep -a "노드 힘 검산\|장면\]" $O/cover/cover.log | cut -c1-400; cat $O/cover/cover.txt
  grep -q MOM_COVER_DONE $O/cover/cover.log || fail "cover 실패 $(grep -a 'Traceback\|Error' $O/cover/cover.log | tail -2)"
  up
  # 2) 이어 돌리기: A = 20 (10 마다 저장) 뒤 30 까지 이어서, B = 한 번에 30 -> 망·Adam·생성기 비교
  if [ ! -f $O/resume.ok ]; then
    RA=$O/resA; RB=$O/resB; rm -rf $RA $RB; mkdir -p $RA $RB
    RS="--mode train --batch 2 --ckpt_every 10 --eval_every 100000 --eval_len 3 --n_stats 8 --log_every 5"
    CUDA_VISIBLE_DEVICES=$G $TM $RS --iters 20 --out $RA > $RA/a20.log 2>&1 < /dev/null
    CUDA_VISIBLE_DEVICES=$G $TM $RS --iters 30 --out $RA > $RA/a30.log 2>&1 < /dev/null
    CUDA_VISIBLE_DEVICES=$G $TM $RS --iters 30 --out $RB > $RB/b30.log 2>&1 < /dev/null
    CMP=$(python - $RA/ck.pt $RB/ck.pt <<'PY'
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False); b = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
d = max(float((a["net"][k].float() - b["net"][k].float()).abs().max()) for k in a["net"])
s = max(float(b["net"][k].float().abs().max()) for k in b["net"])
o = max(float((x - y).abs().max()) for x, y in zip([t for st in a["opt"]["state"].values() for t in st.values() if torch.is_tensor(t) and t.numel() > 1],
                                                    [t for st in b["opt"]["state"].values() for t in st.values() if torch.is_tensor(t) and t.numel() > 1]))
print(f"반복 {a['it']}/{b['it']} 망 파라미터 최대차 {d:.3e} (최대 크기 {s:.3e}) Adam 상태 최대차 {o:.3e} 생성기 같음 {bool((a['rng']['gen'] == b['rng']['gen']).all())} "
      f"평가 기록 {len(a['hist'])}/{len(b['hist'])}")
PY
)
    RSL=$(grep -a -m1 "이어 돈다" $RA/a30.log | cut -c1-120); TB_=$(cat $RA/*.log $RB/*.log | grep -a -c Traceback)
    echo "[이어 돌리기] $RSL | $CMP | Traceback $TB_" | tee $O/resume_test.txt
    grep -q "반복 30/30" <<<"$CMP" && grep -q "생성기 같음 True" <<<"$CMP" && [ -n "$RSL" ] && [ "$TB_" = 0 ] || fail "이어 돌리기 확인 실패"
    python -c "import re,sys;m=re.search(r'망 파라미터 최대차 ([0-9.e+-]+)', sys.argv[1]);sys.exit(0 if m and float(m.group(1))<1e-3 else 1)" "$CMP" || fail "이어 돈 것과 한 번에 돈 것의 망 차이 1e-3 이상"
    touch $O/resume.ok; up
  fi
  # 3) 짧은 학습 + 롤아웃·렌더 시험
  SM=$O/smoke; mkdir -p $SM; NS=${MOM_SMOKE_IT:-300}
  if [ "$(ck_it $SM/ck.pt)" != "$NS" ]; then
    echo "[짧은 학습 명령] CUDA_VISIBLE_DEVICES=$G $TM --mode train --iters $NS --eval_every 100 --ckpt_every 100 --log_every 10 --out $SM --r2 $RO/smoke" > $SM/cmd.txt
    CUDA_VISIBLE_DEVICES=$G $TM --mode train --iters $NS --eval_every 100 --ckpt_every 100 --log_every 10 --out $SM --r2 $RO/smoke >> $SM/train.log 2>&1 < /dev/null
  fi
  grep -a "\[반복\|\[평가\]\|입력 표준화\|Traceback\|Error" $SM/train.log | cut -c1-300 | tail -20
  [ "$(ck_it $SM/ck.pt)" = "$NS" ] || fail "짧은 학습 실패"
  render $SM/ck_best.pt $SM/roll "변위 망 (운동량 잔차 학습) 시험 $NS 회" || fail "롤아웃·렌더 시험 실패"
  touch $O/MOM_TEST_DONE; up; echo MOM_TEST_DONE; exit 0
fi

# ---------------- run: 본 학습 ----------------
NIT=${MOM_ITERS:-30000}
K=$(ck_it $O/ck.pt)
if [ "$K" != "$NIT" ]; then
  CMD="CUDA_VISIBLE_DEVICES=$G $TM --mode train --iters $NIT --batch ${MOM_BATCH:-4} --eval_every 1000 --ckpt_every 500 --out $O --r2 $RO"
  echo "[학습 시작] 커밋 $C GPU$G 이어 돌 반복 ${K:-없음} $(date '+%F %T')" | tee -a $O/${MK}_LAUNCHED
  echo "[학습 명령줄] $CMD" | tee -a $O/${MK}_LAUNCHED; up
  eval "$CMD" >> $O/train.log 2>&1 < /dev/null
  K=$(ck_it $O/ck.pt); echo "[학습 끝] 반복 $K $(date +%T)"; up
  [ "$K" = "$NIT" ] || fail "학습이 $NIT 에 못 미침 (반복 $K) $(grep -a 'Traceback\|Error' $O/train.log | tail -2)"
fi
render $O/ck_best.pt $O/final "변위 망 (운동량 잔차 학습)" || fail "최종 롤아웃·렌더 실패"
python - $O/final/summary_mom.json <<'PY' | tee $O/err_cmp.txt
import json, sys
s = json.load(open(sys.argv[1]))
print(f"[오차 비교] lego L000 80 프레임 (h = 1/60, 롤아웃 평균 |x − x_gt| / EXT)")
print(f"  변위 망 (운동량 잔차 학습, 반복 {s['it']} 최저 체크포인트): {100 * s['err_mean']:.3f}% | 뒤집힌 사면체 합 {s['inv_tets_sum']} 최대 {s['inv_tets_max']} | "
      f"뒤집힌 입자 최대 {s['inv_particles_max']} | 프레임당 {s['sec_per_frame']:.4f} s (중앙 {s['sec_per_frame_med']:.4f}) | 발산 {s['diverged_at']}")
print("  PCG FEM (final3_eig32, 단독): 11.30% (수렴판 11.304%), 0.731 s/프레임")
print("  Graph U-Net 풀 학습 (pool1_gunet): 22.85%")
PY
kill $SYNC 2>/dev/null
echo "${MK}_DONE $(date) 커밋 $C $(head -2 $O/err_cmp.txt | tail -1)" > $O/${MK}_DONE; up
rclone check $O $RO --one-way --size-only -q && echo "[R2 대조] 전부 올라감"
echo "${MK}_DONE $(date)"
