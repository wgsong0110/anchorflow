#!/bin/bash
# lego 새 바닥(softplus + 미끄러짐 방지) 다섯 궤적 상태 적합의 "공중 상태 기울기가 접촉보다 훨씬 작다" 가설 진단 (사용자 10-21).
# 학습(spstick.sh multi, 커밋 13471f4)과 같은 BASE·같은 AF_STATE_FIT_MULTI 경로로, 400 상태를 하나씩 (exe/train_deform.py AF_GRADAC 훅):
#   (a) 학습 끝 망 lbours/spstick/multi/ck.pt (+ AF_GRADAC_U: u=0, u=u₀ 출발점의 ∂E/∂u)  (b) 같은 설정·시드 새 망(학습 시작 시점)
# 먼저 (a) 2 상태 시험 -> 통과하면 (a)(b) 두 GPU 동시. 출력은 처음부터 lbours/grad_air_contact 와 R2 통째 동기화(제외 없음).
# 잡 안에서만: AF_DETACH=1 bash exe/job_run.sh <이 파일> <로그>.  끝 표식: 로그 GRADAC_DONE / GRADAC_FAIL
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
# 워크트리는 부트스트랩($W/gradac_boot.sh)이 이 커밋으로 만들어 두고 그 안의 이 파일을 부른다
cd "$(dirname "$0")/.." || exit 1
P=$W/lbpool; SPU=$W/lbours/spstick; U=$W/lbours/grad_air_contact; R=r2:storage/result/anchorflow/lbours/grad_air_contact; mkdir -p $U
echo "[시작] $(date) 커밋 $(git rev-parse --short HEAD) $(hostname)"
# 학습 실행 로그(spstick.out)의 [BASE] 줄 그대로
BASE="--no_mat --arch sgnn --gnn_layers 8 --hidden 128 --obj pts --net_inertial --bc_level node --bc_mode project --bc_softplus 0.01 --inv_barrier 0 --node_h 0.028284 --dt_cond --dt_scale --v0_traj --det_eps 0.1 --det_w 0 --det_keep_elastic --lr 3e-4 --batch 8 --det_every 200 --pool_fill $P/fill --pool_cfg $P/cfg --pool_combos lego_jelly --pool_drop 0.0,0.02,0.0,1.5 --pool_frames 80 --phase2 --phys_w 1.0 --save_every 100000 --bc_stick 0.01 --bc_stick_xn"
[ "[BASE] $BASE" = "$(grep -a -m1 '^\[BASE\]' $W/spstick.out)" ] && echo "[BASE] 학습 실행 로그와 일치" || { echo "[BASE] 학습 로그와 다름 -- 중단 GRADAC_FAIL"; exit 1; }
SPEC=""; for T in L000 0000 0001 0002 0003; do SPEC="$SPEC${SPEC:+,}$T=$SPU/roll_ov_$T.pt"; done
NOCK=$U/no_ckpt_init.pt; [ -e $NOCK ] && { echo "$NOCK 가 있다 -- 새 망이 아니게 된다 GRADAC_FAIL"; exit 1; }
( while true; do rclone copy $U $R -q; sleep 120; done ) & SYNC=$!
fin(){ kill $SYNC 2>/dev/null; rclone copy $U $R -q; echo "[R2] $(rclone size $R 2>&1 | tr '\n' ' ')"; echo "$1"; exit 0; }
GP=($(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2<1024{print $1}'))
echo "[GPU] 빈 카드: ${GP[*]}"; [ ${#GP[@]} -ge 2 ] || fin "빈 GPU 부족 GRADAC_FAIL"
run(){ nm=$1; g=$2; ck=$3; shift 3
  env "$@" AF_STATE_FIT_MULTI="$SPEC:0:8" AF_STATE_FIT_LOSS=ip AF_STATE_FIT_CK=$ck AF_STATE_FIT_TB=$U/tb_dummy_$nm \
    CUDA_VISIBLE_DEVICES=$g python -u exe/train_deform.py --infer --data $W/multi5 --hold_traj L000,0000,0001,0002,0003 \
    --out $U/run_$nm --tag EV $BASE --iters 0 --eval_t0 0 --eval_len 80 --tb none > $U/log_$nm.txt 2>&1 < /dev/null; }
if ! grep -q GRADAC_PROC_DONE $U/log_smoke.txt 2>/dev/null; then
  run smoke ${GP[0]} $SPU/multi/ck.pt AF_GRADAC=$U/states_smoke.jsonl AF_GRADAC_U=1 AF_GRADAC_N=2
  grep -a "기울기 진단\|이어 돈다\|Traceback\|Error" $U/log_smoke.txt | tail -5 | cut -c1-250
  grep -q GRADAC_PROC_DONE $U/log_smoke.txt || { tail -30 $U/log_smoke.txt; fin "시험 실패 GRADAC_FAIL"; }
fi
head -c 1500 $U/states_smoke.jsonl; echo
run trained ${GP[0]} $SPU/multi/ck.pt AF_GRADAC=$U/states_trained.jsonl AF_GRADAC_U=1 & PA=$!
run init ${GP[1]} $NOCK AF_GRADAC=$U/states_init.jsonl & PB=$!
wait $PA; wait $PB
for nm in trained init; do grep -a "기울기 진단\|이어 돈다\|새 망\|Traceback\|Error" $U/log_$nm.txt | tail -4 | cut -c1-250; done
[ -e $NOCK ] && echo "[경고] $NOCK 가 생겼다"
if grep -q GRADAC_PROC_DONE $U/log_trained.txt && grep -q GRADAC_PROC_DONE $U/log_init.txt; then
  echo "[줄 수] $(wc -l < $U/states_trained.jsonl) / $(wc -l < $U/states_init.jsonl)"; fin "GRADAC_DONE"
else fin "GRADAC_FAIL"; fi
