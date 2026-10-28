#!/bin/bash
# ficus FEM 실시간 판 (사용자 10-28 "lego 입자 바닥 + 혼합 정밀도 PCG 실시간 판과 같은 설정으로 ficus"):
#   풀이 설정 = lego i_tmix_asm_r3 cmd.txt 그대로 (사면체 적분점, 뉴턴 1 걸음, 타일 혼합 정밀도 PCG δ 0.1 rtol 1e-6, 조립 커널, 커스텀 CUDA 커널),
#   물리 조건 = 기존 ficus FEM 실행 (lbours/fem*_ficus: train_deform --cfg_bc --pg_region_density --pg_vdamp, 중력 0, 바닥 없음, --node_h 0.056569,
#   E 2e6 ν 0.4, h = frame_dt 0.04) 과 같은 정의 -> fem_rt_sim --floor none (바닥 항 꺼짐; "입자 바닥" 해당 없음) --cfg_bc --region_density --pg_vdamp.
# 단계 (한 GPU 에 순차, 측정은 노드에 이것 하나):
#   1. train_deform 사면체 적분점 경로 (AF_FEM_TETQ=1, 뉴턴 1) 3 프레임 + 프레임 0·1·2 첫 걸음 덤프 (AF_FEM_TETQ_DUMP)
#   2. fem_rt 3 프레임 (같은 상태에서 이어 비교할 위치)   3. exe/fem_rt_check_ficus.py 대조 (E·기울기·헤시안·방향, 고정 노드·w·질량·감쇠·U0)
#   4. 본 실행 125 프레임 (F float32 덤프)   5. 오차 지표 재현 (기존 세 판 roll_fem.pt 를 같은 식으로)   6. 3DGS 렌더   7. ψ (기존 파이프라인, 고정 범위)
# 사용 (잡 안에서만): AF_DETACH=1 AF_JOB=.. AF_NODE=.. bash exe/job_run.sh <작업트리>/exe/fem_rt_ficus.sh <로그>     표식 lbours/fem_rt_ficus/FICUS_RT_DONE
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD); G=${G:-0}
U=$W/lbours/fem_rt_ficus; R=r2:storage/result/anchorflow/lbours/fem_rt_ficus; mkdir -p $U/chk
cp "$0" $U/fem_rt_ficus.sh 2>/dev/null
( while true; do sleep 120; rclone copy $U $R -q; done ) & SYNC=$!
trap 'kill $SYNC 2>/dev/null' EXIT
DATA=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt; GTD=$W/lbours/ficus_v0/gt; TOPO=$W/topo/pg_ficus_elastic_v0
SOLVE="--fast --cuda --graph --lite --warp --near32 --sym --true_res --asm_k --pcg2 6 --mix_delta 0.1"
PHYS="--data $DATA --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
TG=rt_tmix; O=$U/$TG
echo "[시작] $(date) 커밋 $C 노드 $(hostname) 잡 ${PBS_JOBID:-?} GPU$G"
# 1. train_deform 사면체 적분점 3 프레임 + 덤프
TDO=$W/lbours/femrtchk_ficus/short_$C
if [ ! -f $U/chk/TD_DONE ]; then
  env AF_FEM_TETQ=1 AF_FEM_FAST_K=manual AF_FEM_FAST_NEAR=analytic AF_FEM_EIG_F32=1 AF_FEM_TETQ_DUMP=$U/chk/dump AF_FEM_TETQ_DUMP_T=0,1,2 \
    FEM_SOLVER=pcg FEM_PCG_MAX=10000 FEM_EIG_FLOOR=1e-6 FEM_LM=1 FEM_LM_MUMIN=0 FEM_EIG_PROJ=abs FEM_FIX_NEWTON=1 FEM_FAST=tet FEM_FAST_CHECK=0 \
    FEM_PCG_CHECK=50 FEM_CSR_CACHE=1 FEM_ASM_CHECK=0 FEM_OUT=femrtchk FEM_SHORT=3 bash exe/fem_roll.sh ficus $G short < /dev/null > $U/chk/td_short.out 2>&1
  if [ -f $TDO/roll_fem.pt ] && [ -f $U/chk/dump/tetq_t0002.pt ]; then touch $U/chk/TD_DONE
  else echo "[오류] train_deform 대조 실행 실패"; tail -5 $U/chk/td_short.out; grep -a "Traceback\|Error" $TDO/v_fem.log | tail -3; fi
  cp $TDO/cmd.txt $U/chk/td_cmd.txt 2>/dev/null; echo "AF_FEM_TETQ=1 AF_FEM_FAST_K=manual AF_FEM_FAST_NEAR=analytic AF_FEM_EIG_F32=1 (환경으로 추가)" >> $U/chk/td_cmd.txt
fi
# 2. fem_rt 3 프레임
[ -f $U/chk3/RT_DONE ] || RT_ROOT=fem_rt_ficus bash exe/fem_rt_sim.sh chk3 $G $PHYS $SOLVE --frames 3 --no_dump --dump_pred < /dev/null > $U/chk/chk3.out 2>&1
# 3. 대조
if [ -f $U/chk/TD_DONE ] && ! grep -q CHECK_DONE $U/chk/check.log 2>/dev/null; then
  CUDA_VISIBLE_DEVICES=$G python -u exe/fem_rt_check_ficus.py --dump $U/chk/dump --t 0,1,2 --data $DATA --out $U/chk/check.json < /dev/null > $U/chk/check.log 2>&1
fi
python - $TDO/roll_fem.pt $U/chk3/pred.pt <<'PY' > $U/chk/traj3.txt 2>&1
import sys, torch
A = torch.load(sys.argv[1], map_location="cpu", weights_only=False); B = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
ext = float(A["EXT"]); print(f"[3 프레임 위치 대조] train_deform 사면체 적분점 (뉴턴 1) vs fem_rt, EXT {ext:.6f} (fem_rt EXT {float(B['EXT']):.6f})")
for i in range(min(A["pred"].shape[0], B["pred"].shape[0])):
    a, b, g = A["pred"][i].float(), B["pred"][i].float(), A["gt"][i].float()
    d = (a - b).norm(dim=-1) / ext
    print(f"  프레임 {i + 1}: |Δx|/EXT 평균 {d.mean():.3e} 최대 {d.max():.3e} | 정답 오차 train_deform {100 * float((a - g).norm(dim=-1).mean()) / ext:.4f}% fem_rt {100 * float((b - g).norm(dim=-1).mean()) / ext:.4f}% | 이동 평균 {float((a - A['x0'].float()).norm(dim=-1).mean()) / ext:.3e}")
PY
cat $U/chk/traj3.txt; tail -2 $U/chk/check.log
# 4. 본 실행 (노드 7 장 모두 빌 때까지 기다린 뒤 -- 단독 측정)
if [ ! -f $O/RT_DONE ]; then
  while [ -n "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 500')" ]; do echo "[대기] 다른 GPU 사용 중 $(date +%T)"; sleep 60; done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $U/gpus_before_main.txt
  RT_ROOT=fem_rt_ficus bash exe/fem_rt_sim.sh $TG $G $PHYS $SOLVE --dump_f32 < /dev/null > $U/main.out 2>&1
fi
[ -f $O/RT_DONE ] || { echo "[오류] 본 실행 실패 $(cat $O/RT_FAIL 2>/dev/null)"; rclone copy $U $R -q; echo FICUS_RT_FAIL; exit 1; }
cat $O/err_fem.txt
# 5. 오차 지표 재현 (같은 식: 프레임마다 입자 평균 |x - x_gt| / EXT 의 프레임 평균)
python - $W/lbours/fem_ficus/run/roll_fem.pt $W/lbours/fem2_ficus/run/roll_fem.pt $W/lbours/fem3_ficus/run/roll_fem.pt $O/roll_fem.pt <<'PY' > $U/err_same_metric.txt 2>&1
import sys, torch
for p in sys.argv[1:]:
    D = torch.load(p, map_location="cpu", weights_only=False)
    P, Gt, ext = D["pred"].float(), D["gt"].float(), float(D["EXT"])
    e = torch.stack([(P[i] - Gt[i]).norm(dim=-1).mean() for i in range(P.shape[0])]) / ext
    bb = float((P[-1].max(0).values - P[-1].min(0).values).norm()) / ext
    print(f"{p}: 프레임 {P.shape[0]}, 오차 {100 * float(e.mean()):.4f}% (EXT {ext:.6f}), 마지막 프레임 바운딩박스 대각/EXT {bb:.4f}")
PY
cat $U/err_same_metric.txt
rclone copy $U $R -q
# 6. 3DGS 렌더 (설명 글자는 비우고 나중에 상단 띠로 합성; 프레임 번호는 렌더에)
[ -f $O/h5_fem/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_fem.pt --traj $GTD --out $O/h5_fem > $O/h5_fem.log 2>&1
mkdir -p $O/v
[ -f $O/v/fem_ficus.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_fem --config $GTD/config.json --shape ficus --out $O/v/fem_ficus.mp4 \
   --frame_label 0 --caption " " < /dev/null > $O/v/r_fem.log 2>&1
[ -f $O/v/fem_ficus.mp4 ] || echo "[오류] 렌더 실패 $(tail -3 $O/v/r_fem.log)"
# 7. ψ 색 (fem3_ficus_boot.sh 와 같은 파이프라인·같은 고정 범위)
LO=-2.9999966621398926; HI=4.933352713584899; D=$O/psi; mkdir -p $D
grep -q PSI_COLOR_DONE $D/psi_color.log 2>/dev/null || CUDA_VISIBLE_DEVICES=$G python -u exe/psi_color.py --gt_h5 $TOPO --dump $O/roll_fem.pt \
    --config $GTD/config.json --out $D --eps 1e-3 --q 1 99 --range $LO $HI --cmap turbo --stats_frames 1 40 125 > $D/psi_color.log 2>&1 < /dev/null
grep -a "색 범위\|변형\]" $D/psi_color.log
[ -f $D/psi_fem_ficus.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_fem --config $GTD/config.json --shape ficus \
    --colors $D/col_ov.npy --out $D/psi_fem_ficus.mp4 > $D/r_fem.log 2>&1 < /dev/null
python - $D/psi_gt.npy $W/lbours/fem_ficus/psi/cg/psi_gt.npy $D/col_gt.npy $W/lbours/fem_ficus/psi/cg/col_gt.npy <<'PY' | tee $D/gt_same.txt
import sys, numpy as np
a, b, c, d = (np.load(p) for p in sys.argv[1:5])
print(f"정답 psi_gt·col_gt 기존 fem_ficus/psi/cg 와 비트 동일 = {int(np.array_equal(a, b) and np.array_equal(c, d))}")
PY
kill $SYNC 2>/dev/null
rclone copy $U $R -q
if [ -f $O/v/fem_ficus.mp4 ] && [ -f $D/psi_fem_ficus.mp4 ]; then touch $U/FICUS_RT_DONE; echo FICUS_RT_DONE; else echo FICUS_RT_PARTIAL; fi
rclone copy $U $R -q
echo "[끝] $(date)"
