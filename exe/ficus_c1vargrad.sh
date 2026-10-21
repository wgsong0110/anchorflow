#!/bin/bash
# ficus C1 N1 (12.98%) 최적화 변수 θ 기준 기울기 (사용자 10-21): AF_VARGRAD.
#  스텝 t=0..124 (프레임 t+1): 시작점 g0, 덤프 역변환 해 θ* 의 g·변수 종류별 노름·뉴턴 감소량, 덤프 프레임 t+1 검산.
#  C1 N1(c1_base pl) / 선형 ficus_v0 (다시 풀어 종료 통계) / c1_base none 첫 스텝 500 회 (다시 풀기).
#  MODE=test 이면 3 스텝만. 출력 lbours/ficus_c1/vargrad[/test], R2 통째.
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODE=${MODE:-full}; GA=${GA:-3}; GB=${GB:-5}; GC=${GC:-6}
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
C=eb533ff; WT=$W/wt/vargrad_$C
[ -d $WT ] || (cd $W/anchorflow && git fetch -q origin && git worktree add -q --detach $WT $C) || { echo "[오류] worktree"; exit 1; }
O0=$W/lbours/ficus_v0; OC=$W/lbours/ficus_c1; OD=$OC/vargrad; R=r2:storage/result/anchorflow/lbours/ficus_c1/vargrad
[ $MODE = test ] && { OD=$OD/test; R=$R/test; }
T=F_pg_v0; DATA=$O0/data/$T; mkdir -p $OD; cd $WT || exit 1
echo "[시작] $(date) 커밋 $(git rev-parse --short HEAD) 모드 $MODE GPU $GA $GB $GC"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
( while true; do rclone copy $OD $R -q; sleep 300; done ) & SYNC=$!
COMMON="--no_mat --arch sgnn --gnn_layers 8 --hidden 128 --obj pts --net_inertial"
FLOOR="--bc_level node --bc_mode project"
REST="--inv_barrier 0 --node_h 0.056569 --dt_cond --dt_scale --v0_traj --det_eps 0.1 --det_w 0 --det_keep_elastic --lr 3e-4 --batch 8 --det_every 200 --pool_fill $W/lbpool/fill --pool_cfg $W/lbpool/cfg --pool_combos lego_jelly --pool_drop 0.0,0.02,0.0,1.5 --pool_frames 80 --phase2 --phys_w 1.0 --save_every 100000"
PGF="--cfg_bc --pg_region_density --pg_vdamp"
BASE="$COMMON $FLOOR $REST $PGF"
OV="--out_var --ov_init predict --ov_roll 500 --ov_opt lbfgs --ov_vars out --ov_obj ip"
C1="--interp c1wf --c1_barrier 1.0 --c1_jhat 0.3"
LEN=125; [ $MODE = test ] && LEN=3
run(){  # 이름 GPU 길이 추가환경 추가인자
  local O=$OD/$1; mkdir -p $O; rm -f $O/vargrad.jsonl $O/vg_t*.pt
  echo "[$1 시작] $(date +%T) GPU $2"
  env $4 AF_VARGRAD=$O AF_VARGRAD_CG=100 CUDA_VISIBLE_DEVICES=$2 \
    python -u exe/train_deform.py --infer --data $DATA --hold_traj $T --out $O/ov --tag OVN $BASE $OV $5 --iters 0 --eval_t0 0 --eval_len $3 --tb none < /dev/null > $O/v.log 2>&1
  grep -a "Traceback\|Error" $O/v.log | head -3
  echo "[$1 끝] $(date +%T) 스텝 $(wc -l < $O/vargrad.jsonl)"; touch $O/DONE
}
( run c1 $GA $LEN "AF_OV_NODE_TGT=$OC/N1/roll_ov.pt" "$C1" ) & P1=$!
( run bary $GB $LEN "AF_OV_NODE_TGT=$O0/roll_ov.pt AF_VARGRAD_RESOLVE=1" "" ) & P2=$!
( run basenone $GC 1 "AF_VARGRAD_RESOLVE=1" "$C1 --c1_base none" ) & P3=$!
wait $P1 $P2 $P3
kill $SYNC; rclone copy $OD $R -q
ok=1; for n in c1 bary basenone; do [ -s $OD/$n/vargrad.jsonl ] || ok=0; done
[ $ok = 1 ] && touch $OD/C1VARGRAD_DONE || touch $OD/C1VARGRAD_FAIL
rclone copy $OD $R -q; ls $OD; echo "[전체 끝] $(date)"
