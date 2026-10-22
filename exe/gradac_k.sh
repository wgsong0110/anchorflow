#!/bin/bash
# 기울기 진단 (b) 행 대체: 학습 초기 반복 K 시점 망 (사용자 10-22). 원래 학습 ck 는 500 마다 덮어써져 초기 것이 없다 ->
# spstick.sh multi 와 **같은 명령·시드·커밋 13471f4**(워크트리 wt/spstick_13471f4)로 처음부터 K 반복까지만 다시 학습해 ck_K 저장
# (AF_STATE_FIT_MULTI 반복 수만 K), 그 망으로 gradac 와 같은 측정(AF_GRADAC, 커밋 gradac 워크트리 = 13471f4 + 측정 훅).
# K: TB 공중 상태(0000:0–11, 0002:0–14, 0003:0–4) 손실이 가장 큰 반복 1, 후보 2·5·100·500·2000. GPU 0 제외, 빈 카드만.
# 출력 lbours/grad_air_contact/k 와 R2 통째 동기화. 끝 표식: 로그 GRADACK_DONE / GRADACK_FAIL
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
GWT=$(cd "$(dirname "$0")/.." && pwd); TWT=$W/wt/spstick_13471f4
[ "$(git -C $TWT rev-parse --short HEAD)" = 13471f4 ] || { echo "학습 워크트리 커밋 다름 GRADACK_FAIL"; exit 1; }
[ -z "$(git -C $TWT status --porcelain --untracked-files=no)" ] || { echo "학습 워크트리 수정됨 GRADACK_FAIL"; exit 1; }
P=$W/lbpool; SPU=$W/lbours/spstick; U=$W/lbours/grad_air_contact/k; R=r2:storage/result/anchorflow/lbours/grad_air_contact/k; mkdir -p $U
echo "[시작] $(date) 측정 커밋 $(git -C $GWT rev-parse --short HEAD), 학습 커밋 13471f4 $(hostname)"
BASE="--no_mat --arch sgnn --gnn_layers 8 --hidden 128 --obj pts --net_inertial --bc_level node --bc_mode project --bc_softplus 0.01 --inv_barrier 0 --node_h 0.028284 --dt_cond --dt_scale --v0_traj --det_eps 0.1 --det_w 0 --det_keep_elastic --lr 3e-4 --batch 8 --det_every 200 --pool_fill $P/fill --pool_cfg $P/cfg --pool_combos lego_jelly --pool_drop 0.0,0.02,0.0,1.5 --pool_frames 80 --phase2 --phys_w 1.0 --save_every 100000 --bc_stick 0.01 --bc_stick_xn"
[ "[BASE] $BASE" = "$(grep -a -m1 '^\[BASE\]' $W/spstick.out)" ] && echo "[BASE] 학습 실행 로그와 일치" || { echo "[BASE] 다름 GRADACK_FAIL"; exit 1; }
SPEC=""; for T in L000 0000 0001 0002 0003; do SPEC="$SPEC${SPEC:+,}$T=$SPU/roll_ov_$T.pt"; done
( while true; do rclone copy $U $R -q 2>/dev/null; sleep 120; done ) & SYNC=$!
GP=($(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$1!=0 && $2<1024{print $1}'))
echo "[GPU] 빈 카드(0 제외): ${GP[*]}"; [ ${#GP[@]} -ge 3 ] || { kill $SYNC; echo "빈 GPU 부족 GRADACK_FAIL"; exit 0; }
cmd(){ # $1 작업폴더 $2 GPU $3 반복수 $4 ck $5 tb $6 로그, 나머지 env
  wt=$1 g=$2 n=$3 ck=$4 tb=$5 lg=$6; shift 6
  ( cd $wt && env "$@" AF_STATE_FIT_MULTI="$SPEC:$n:8" AF_STATE_FIT_LOSS=ip AF_STATE_FIT_CK=$ck AF_STATE_FIT_TB=$tb \
    CUDA_VISIBLE_DEVICES=$g python -u exe/train_deform.py --infer --data $W/multi5 --hold_traj L000,0000,0001,0002,0003 \
    --out $U/run_$(basename $lg .txt) --tag EV $BASE --iters 0 --eval_t0 0 --eval_len 80 --tb none > $lg 2>&1 < /dev/null ); }
one(){ K=$1; g=$2
  if [ ! -f $U/ck_$K.pt ]; then
    rm -rf $U/tb_train_$K; echo "[학습] K=$K GPU$g $(date +%T)"
    cmd $TWT $g $K $U/ck_$K.pt $U/tb_train_$K $U/log_train_$K.txt
    grep -a "새 망\|이어 돈다\|Traceback\|Error" $U/log_train_$K.txt | tail -3 | cut -c1-200
  fi
  KK=$(python -c "import torch;print(torch.load('$U/ck_$K.pt',map_location='cpu',weights_only=False)['k'])" 2>/dev/null)
  [ "$KK" = "$K" ] || { echo "[오류] K=$K 체크포인트 반복 $KK"; return 1; }
  rclone copyto $U/ck_$K.pt $R/ck_$K.pt -q
  if ! grep -q GRADAC_PROC_DONE $U/log_meas_$K.txt 2>/dev/null; then
    echo "[측정] K=$K GPU$g $(date +%T)"
    cmd $GWT $g 0 $U/ck_$K.pt $U/tb_dummy_$K $U/log_meas_$K.txt AF_GRADAC=$U/states_k$K.jsonl
  fi
  grep -a "기울기 진단\] \|기울기 진단 끝\|Traceback" $U/log_meas_$K.txt | tail -2 | cut -c1-200
  grep -q GRADAC_PROC_DONE $U/log_meas_$K.txt; }
( for K in 1 2 5; do one $K ${GP[0]} || exit 1; done ) > $U/chain_a.txt 2>&1 & A=$!
( for K in 100 500; do one $K ${GP[1]} || exit 1; done ) > $U/chain_b.txt 2>&1 & B=$!
( one 2000 ${GP[2]} ) > $U/chain_c.txt 2>&1 & C=$!
wait $A; ra=$?; wait $B; rb=$?; wait $C; rc=$?
cat $U/chain_a.txt $U/chain_b.txt $U/chain_c.txt
kill $SYNC 2>/dev/null; rclone copy $U $R -q; echo "[R2] $(rclone size $R 2>&1 | tr '\n' ' ')"
[ $ra = 0 ] && [ $rb = 0 ] && [ $rc = 0 ] && echo GRADACK_DONE || echo "GRADACK_FAIL ($ra $rb $rc)"
