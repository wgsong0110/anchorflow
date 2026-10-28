#!/bin/bash
# ficus FEM 실시간 판 + 수렴 판정 수정 (사용자 10-28 "해봐"): exe/fem_rt_ficus.sh 의 rt_tmix 와 같은 풀이·물리 조건 +
#   --min_steps 1 --gtol 1e-3 (train_deform AF_FEM_MIN_STEPS=1 AF_FEM_GTOL=1e-3 와 같은 정의, fem3_ficus 와 같다).
#   뉴턴 걸음 상한은 그대로 1 -> 판정은 걸음 0 에서만 일어나고, 그때 받아들인 걸음 0 개·|g|/|g0| = 1 이라 둘 다 걸음 0 수렴을 막는다
#   = 매 프레임 뉴턴 1 걸음 (선탐색·LM 그대로) 을 반드시 받는다. gtol 은 상한 1 에서는 효과가 없다 (기록용).
# 그리고 ψ 띠 원인 확인: 같은 정답 시작 상태에서 사면체 적분점 수렴 판 (뉴턴 상한 100, 같은 판정) 3 프레임 + exe/fem_rt_psi_band.py.
# 사용 (잡 안에서만): AF_DETACH=1 bash exe/job_run.sh <작업트리>/exe/fem_rt_ficus_fix.sh <로그>     표식 lbours/fem_rt_ficus/FICUS_RT_FIX_DONE
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD); G=${G:-0}
U=$W/lbours/fem_rt_ficus; R=r2:storage/result/anchorflow/lbours/fem_rt_ficus; mkdir -p $U/band
cp "$0" $U/fem_rt_ficus_fix.sh 2>/dev/null
( while true; do sleep 120; rclone copy $U $R -q; done ) & SYNC=$!
trap 'kill $SYNC 2>/dev/null' EXIT
DATA=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt; GTD=$W/lbours/ficus_v0/gt; TOPO=$W/topo/pg_ficus_elastic_v0
SOLVE="--fast --cuda --graph --lite --warp --near32 --sym --true_res --asm_k --pcg2 6 --mix_delta 0.1"
PHYS="--data $DATA --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp"
JUDGE="--min_steps 1 --gtol 1e-3"
TG=rt_tmix_fix; O=$U/$TG
echo "[시작] $(date) 커밋 $C 노드 $(hostname) GPU$G"
# 1. 본 실행 (노드 7 장 모두 빌 때까지 기다린 뒤 -- 단독 측정)
if [ ! -f $O/RT_DONE ]; then
  while [ -n "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 500')" ]; do echo "[대기] 다른 GPU 사용 중 $(date +%T)"; sleep 60; done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $U/gpus_before_fix.txt
  RT_ROOT=fem_rt_ficus bash exe/fem_rt_sim.sh $TG $G $PHYS $SOLVE $JUDGE --dump_f32 < /dev/null > $U/fix.out 2>&1
fi
[ -f $O/RT_DONE ] || { echo "[오류] 본 실행 실패 $(cat $O/RT_FAIL 2>/dev/null)"; rclone copy $U $R -q; echo FICUS_RT_FIX_FAIL; exit 1; }
cat $O/err_fem.txt
# 2. 사면체 적분점 수렴 판 3 프레임 (원인 확인용, 뉴턴 상한 100 = train_deform AF_OV_NEWTON_STEPS 100)
[ -f $U/conv3/RT_DONE ] || RT_ROOT=fem_rt_ficus bash exe/fem_rt_sim.sh conv3 $G $PHYS $SOLVE $JUDGE --newton 100 --frames 3 --dump_f32 < /dev/null > $U/band/conv3.out 2>&1
tail -2 $U/band/conv3.out
# 3. 오차 지표 (같은 식)
python - $O/roll_fem.pt <<'PY' > $U/err_fix_same_metric.txt 2>&1
import sys, torch
D = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
P, Gt, ext = D["pred"].float(), D["gt"].float(), float(D["EXT"])
e = torch.stack([(P[i] - Gt[i]).norm(dim=-1).mean() for i in range(P.shape[0])]) / ext
print(f"{sys.argv[1]}: 프레임 {P.shape[0]}, 오차 {100 * float(e.mean()):.4f}% (EXT {ext:.6f}), 마지막 바운딩박스 대각/EXT {float((P[-1].max(0).values - P[-1].min(0).values).norm()) / ext:.4f}")
PY
cat $U/err_fix_same_metric.txt
# 4. ψ 띠 원인
grep -q BAND_DONE $U/band/band.log 2>/dev/null || CUDA_VISIBLE_DEVICES=$G python -u exe/fem_rt_psi_band.py --data $DATA --t 1,2,3,40,125 --out $U/band/band.json \
   --run "실시간=$U/rt_tmix/roll_fem.pt" --run "실시간+수정판정=$O/roll_fem.pt" --run "사면체 적분점 수렴(3 프레임)=$U/conv3/roll_fem.pt" \
   --run "입자 적분점 수정판정(fem3)=$W/lbours/fem3_ficus/run/roll_fem.pt" --run "입자 적분점 기존 PCG(fem2)=$W/lbours/fem2_ficus/run/roll_fem.pt" \
   < /dev/null > $U/band/band.log 2>&1
tail -3 $U/band/band.log | cut -c1-600
rclone copy $U $R -q
# 5. 렌더·ψ (fem_rt_ficus.sh 와 같은 경로·같은 고정 범위)
[ -f $O/h5_fem/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_fem.pt --traj $GTD --out $O/h5_fem > $O/h5_fem.log 2>&1
mkdir -p $O/v
[ -f $O/v/fem_ficus.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_fem --config $GTD/config.json --shape ficus --out $O/v/fem_ficus.mp4 \
   --frame_label 0 --caption " " < /dev/null > $O/v/r_fem.log 2>&1
LO=-2.9999966621398926; HI=4.933352713584899; D=$O/psi; mkdir -p $D
grep -q PSI_COLOR_DONE $D/psi_color.log 2>/dev/null || CUDA_VISIBLE_DEVICES=$G python -u exe/psi_color.py --gt_h5 $TOPO --dump $O/roll_fem.pt \
    --config $GTD/config.json --out $D --eps 1e-3 --q 1 99 --range $LO $HI --cmap turbo --stats_frames 1 40 125 > $D/psi_color.log 2>&1 < /dev/null
[ -f $D/psi_fem_ficus.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_fem --config $GTD/config.json --shape ficus \
    --colors $D/col_ov.npy --out $D/psi_fem_ficus.mp4 > $D/r_fem.log 2>&1 < /dev/null
kill $SYNC 2>/dev/null
rclone copy $U $R -q
if [ -f $O/v/fem_ficus.mp4 ] && [ -f $D/psi_fem_ficus.mp4 ] && grep -q BAND_DONE $U/band/band.log; then touch $U/FICUS_RT_FIX_DONE; echo FICUS_RT_FIX_DONE; else echo FICUS_RT_FIX_PARTIAL; fi
rclone copy $U $R -q
echo "[끝] $(date)"
