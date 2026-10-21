#!/bin/bash
# 고친 코드(커밋 C, 8f602d1 이후 -- pl_c1_dofs 중복 쓰기 역전파 버그 고침)로 돌린 ficus C¹ N1 (lbours/ficus_c1fix/N1) 의 해에서
# 최적화 변수 θ 기준 기울기 (AF_VARGRAD, 버그판 측정과 같은 설정). 출력 lbours/ficus_c1fix/vargrad/c1, R2 통째.
# 빈 GPU (3·5·6 중 메모리 100 MiB 미만) 하나에서만. 사용: C=<커밋> bash ficus_c1vargrad_fix.sh
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
C=${C:?커밋}; WT=$W/wt/vargradfix_$C
OF=$W/lbours/ficus_c1fix; OD=$OF/vargrad; R=r2:storage/result/anchorflow/lbours/ficus_c1fix/vargrad; mkdir -p $OD
rm -f $OD/C1FIXVG_DONE $OD/C1FIXVG_FAIL
fail(){ echo "[실패] $1"; touch $OD/C1FIXVG_FAIL; rclone copy $OD $R -q; exit 1; }
[ -d $WT ] || (cd $W/anchorflow && git fetch -q origin && git worktree add -q --detach $WT $C) || fail worktree
[ -f $OF/N1/roll_ov.pt ] || fail "덤프 없음 $OF/N1/roll_ov.pt"
G=""; for k in $(seq 1 72); do
  G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '($1==3||$1==5||$1==6)&&$2<100{print $1; exit}')
  [ -n "$G" ] && break; echo "[대기] 빈 GPU 없음 $(date +%T)"; sleep 300
done
[ -n "$G" ] || fail "빈 GPU 없음 6 h"
cd $WT || fail cd
echo "[시작] $(date) 커밋 $(git rev-parse --short HEAD) GPU $G"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
( while true; do rclone copy $OD $R -q; sleep 300; done ) & SYNC=$!
T=F_pg_v0; DATA=$W/lbours/ficus_v0/data/$T
COMMON="--no_mat --arch sgnn --gnn_layers 8 --hidden 128 --obj pts --net_inertial"
FLOOR="--bc_level node --bc_mode project"
REST="--inv_barrier 0 --node_h 0.056569 --dt_cond --dt_scale --v0_traj --det_eps 0.1 --det_w 0 --det_keep_elastic --lr 3e-4 --batch 8 --det_every 200 --pool_fill $W/lbpool/fill --pool_cfg $W/lbpool/cfg --pool_combos lego_jelly --pool_drop 0.0,0.02,0.0,1.5 --pool_frames 80 --phase2 --phys_w 1.0 --save_every 100000"
PGF="--cfg_bc --pg_region_density --pg_vdamp"
OV="--out_var --ov_init predict --ov_roll 500 --ov_opt lbfgs --ov_vars out --ov_obj ip"
C1="--interp c1wf --c1_barrier 1.0 --c1_jhat 0.3"
O=$OD/c1; mkdir -p $O
AF_OV_NODE_TGT=$OF/N1/roll_ov.pt AF_VARGRAD=$O AF_VARGRAD_CG=100 CUDA_VISIBLE_DEVICES=$G \
  python -u exe/train_deform.py --infer --data $DATA --hold_traj $T --out $O/ov --tag OVN $COMMON $FLOOR $REST $PGF $OV $C1 --iters 0 --eval_t0 0 --eval_len 125 --tb none < /dev/null > $O/v.log 2>&1
grep -a "Traceback\|Error" $O/v.log | head -3
n=$(wc -l < $O/vargrad.jsonl 2>/dev/null || echo 0); echo "[끝] $(date +%T) 스텝 $n"
kill $SYNC
[ "$n" -ge 125 ] && touch $OD/C1FIXVG_DONE || touch $OD/C1FIXVG_FAIL
rclone copy $OD $R -q; ls $OD
