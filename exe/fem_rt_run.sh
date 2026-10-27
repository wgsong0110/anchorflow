#!/bin/bash
# FEM pcg 실시간 판 측정 실행 (10-27, doc/fem_realtime.md). exe/fem_roll.sh lego 와 **같은 명령줄·같은 환경변수** (pcg, PCG_MAX 10000,
# EIG_FLOOR 1e-6, LM 1, LM_MUMIN 0, EIG_PROJ abs, tol 1e-2) 에 선택지만 더한다. fem_roll.sh 는 시작 전에 fem2_lego 폴더 통째를
# R2 에 올리느라(동시 7 개) 15 분 넘게 python 을 못 띄워서, 이 드라이버는 python 을 먼저 띄우고 자기 폴더만 백그라운드로 동기화한다.
# 사용: bash exe/fem_rt_run.sh <태그> <GPU> <프레임 수> [이어 돌릴 프레임 s]   (선택지는 환경변수 EXTRA="AF_FEM_FAST=tet ...")
#   s 를 주면 기준 실행 조각(lbours/fem2_lego/run/parts c000{10..s}.pt)에서 s+1 부터 (프레임 수 = 끝 프레임).
# 출력 lbours/fem_rt_lego/<태그>: v_fem.log, fem.jsonl, roll_fem.pt, err_fem.txt, 표식 RT_DONE / RT_FAIL.
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
TG=${1:?태그}; G=${2:?GPU}; L=${3:?프레임 수}; S0=${4:-}
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
O=$W/lbours/fem_rt_lego/$TG; R=r2:storage/result/anchorflow/lbours/fem_rt_lego/$TG; mkdir -p $O
T=L000; DATA=$W/one/L000; NH=0.028284
FLOOR="--bc_level node --bc_mode project --bc_softplus 0.01 --bc_stick 0.01 --bc_stick_xn"
COMMON="--no_mat --arch sgnn --gnn_layers 8 --hidden 128 --obj pts"
REST="--inv_barrier 0 --node_h $NH --dt_cond --dt_scale --v0_traj --det_eps 0.1 --det_w 0 --det_keep_elastic --lr 3e-4 --batch 8 --det_every 200 --pool_fill $W/lbpool/fill --pool_cfg $W/lbpool/cfg --pool_combos lego_jelly --pool_drop 0.0,0.02,0.0,1.5 --pool_frames 80 --phase2 --phys_w 1.0 --save_every 100000"
OV="--out_var --ov_init predict --ov_roll 500 --ov_opt lbfgs --ov_vars out --ov_obj newton"
FEMENV="AF_FEM_TOL=1e-2 AF_OV_NEWTON_STEPS=100 AF_OV_NEWTON_DAMP=0 AF_OV_NEWTON_CG=200 AF_FEM_SOLVER=pcg AF_FEM_ASM_CHECK=0 AF_FEM_PCG_MAX=10000 AF_FEM_PCG_RTOL=1e-6 AF_FEM_EIG_FLOOR=1e-6 AF_FEM_LM=1 AF_FEM_LM_MUMIN=0 AF_FEM_EIG_PROJ=abs"
ARGS="--infer --data $DATA --hold_traj $T --out $O/fem --tag OVN $COMMON $FLOOR $REST $OV --iters 0 --eval_t0 0 --eval_len $L --tb none"
ENVS="$FEMENV ${EXTRA:-} AF_FEM_LOG=$O/fem.jsonl AF_ROLL_DUMP=$O/roll_fem.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 AF_ROLL_PART=$O/parts AF_ROLL_PART_EVERY=10"
if [ -n "$S0" ]; then
  mkdir -p $O/parts0; for n in $(seq 10 10 $S0); do cp -n $W/lbours/fem2_lego/run/parts/c$(printf %05d $n).pt $O/parts0/; done
  ENVS="$ENVS AF_OV_CONT=$O/parts0:$S0"
fi
( while true; do sleep 300; rclone copy $O $R -q; done ) & SYNC=$!
trap 'kill $SYNC 2>/dev/null' EXIT
while [ "$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 500 ]; do echo "[대기] GPU$G $(date +%T)"; sleep 30; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $O/gpu_at_start.txt
echo "[명령] 커밋 $C env $ENVS CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py $ARGS" | tee $O/cmd.txt
env $ENVS CUDA_VISIBLE_DEVICES=$G python -u exe/train_deform.py $ARGS < /dev/null > $O/v_fem.log 2>&1
echo "fem $(grep -a "롤아웃\] $T" $O/v_fem.log | tail -1 | grep -o '평균 [0-9.]*%')" > $O/err_fem.txt
if [ -f $O/roll_fem.pt ]; then touch $O/RT_DONE; else grep -a "Traceback\|Error" $O/v_fem.log | tail -3 > $O/RT_FAIL; fi
cat $O/err_fem.txt
rclone copy $O $R -q
