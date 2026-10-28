#!/bin/bash
# 학습 궤적 0000~0003: 다섯 궤적 모델(spstick/multi/ck.pt) 평가 롤아웃(학습 없음, --iters 0, 노드 덤프·F32) + 렌더(정답·출력만 최적화·모델, spstick.sh 와 같은 lb_render 설정) + 위·아래 꼭지점 비율 영상
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
WS=$W/wt/spstick_13471f4; U=$W/lbours/spstick; P=$W/lbpool; O=$W/lbours/spstick_Fdist/topbot/tr4; R=r2:storage/result/anchorflow/lbours/spstick_Fdist/topbot/tr4; V=r2:storage/result/anchorflow/v20261009
mkdir -p $O; cd $WS || exit 1
echo "[시작] $(date) 커밋 $(git rev-parse --short HEAD)"
( while true; do rclone copy $O $R -q; sleep 300; done ) & SYNC=$!
COMMON="--no_mat --arch sgnn --gnn_layers 8 --hidden 128 --obj pts --net_inertial"
REST_OLD="--inv_barrier 1.0 --inv_jhat 0.3 --inv_ext linear --node_h 0.028284 --dt_cond --dt_scale --v0_traj --det_eps 0.1 --det_w 100 --lr 3e-4 --batch 8 --det_every 200 --pool_fill $P/fill --pool_cfg $P/cfg --pool_combos lego_jelly --pool_drop 0.0,0.02,0.0,1.5 --pool_frames 80 --phase2 --phys_w 1.0 --save_every 100000"
FLOOR_SP="--bc_level node --bc_mode project --bc_softplus 0.01"
REST=$(echo "$REST_OLD" | sed -e 's/--inv_barrier 1.0 --inv_jhat 0.3 --inv_ext linear/--inv_barrier 0/' -e 's/--det_w 100/--det_w 0 --det_keep_elastic/')
BASE="$COMMON $FLOOR_SP $REST --bc_stick 0.01 --bc_stick_xn"
echo "[BASE] $BASE"
echo "[spstick.sh 와 대조] $(grep -c -F 'BASE="$BASE_NOXN --bc_stick_xn"' $W/spstick.sh) $(grep -c -F 'BASE_NOXN="$BASE_SP --bc_stick 0.01"' $W/spstick.sh) $(grep -c -F 'BASE_SP="$COMMON $FLOOR_SP $REST"' $W/spstick.sh) 줄 일치 (3 개 다 1 이어야)"
mapfile -t G < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1!=0 && $2<100{print $1}')
echo "[빈 GPU] ${G[*]}"; [ ${#G[@]} -ge 1 ] || { touch $O/TOPBOT4_FAIL; echo "빈 GPU 없음"; exit 1; }
gtdir(){ python - "$1" <<'PY'
import sys, torch, h5py, numpy as np
T = sys.argv[1]; W = "/home/dkta/work"
g = torch.load(f"{W}/lbours/spstick/roll_ov_{T}.pt", map_location="cpu", weights_only=False, mmap=True)["gt"][0].float().numpy()
for d in (f"{W}/lbdata/test/{T}", f"{W}/lbdata/train/{T}"):  # 덤프 gt 는 half 반올림(실측 4.9e-4) -> 문턱 1e-3
    try:
        x = np.array(h5py.File(f"{d}/sim_0000000001.h5", "r")["x"], dtype=np.float32)
        if x.shape == g.shape and np.abs(x - g).max() < 1e-3: print(d); break
    except Exception: pass
PY
}
job(){ T=$1; g=$2
  [ -f $O/roll_mm_$T.pt ] || AF_ROLL_DUMP=$O/roll_mm_$T.pt AF_ROLL_DUMP_F=32 AF_ROLL_TAG=$T AF_ROLL_T0=0 AF_ROLL_NOGRAD=1 CUDA_VISIBLE_DEVICES=$g \
     python -u exe/train_deform.py --infer --data $W/one/$T --hold_traj $T --out $O/v_mm_$T --tag EV --resume $U/multi/ck.pt $BASE --iters 0 --eval_t0 0 --eval_len 80 --tb none > $O/v_mm_$T.log 2>&1
  [ -f $O/roll_mm_$T.pt ] || { echo "[오류] $T 롤아웃 덤프 없음"; grep -a "Traceback\|Error" $O/v_mm_$T.log | head -3; return 1; }
  echo "[롤아웃 $T] $(grep -a "롤아웃\] $T" $O/v_mm_$T.log | tail -1 | grep -o '평균 [0-9.]*%')"
  TR=$(gtdir $T); [ -n "$TR" ] || { echo "[오류] $T 정답 h5 디렉토리 못 찾음"; return 1; }; echo "[정답 h5 $T] $TR"
  for n in ov mm; do S=$( [ $n = ov ] && echo $U/roll_ov_$T.pt || echo $O/roll_mm_$T.pt )
    [ -f $O/h5_${n}_$T/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $S --traj $TR --out $O/h5_${n}_$T > $O/h5_${n}_$T.log 2>&1; tail -1 $O/h5_${n}_$T.log; done
  for n in gt ov mm; do S=$( [ $n = gt ] && echo $TR || echo $O/h5_${n}_$T )
    [ -f $O/${n}_$T.mp4 ] || CUDA_VISIBLE_DEVICES=$g python -u exe/lb_render.py --sim $S --config $W/lbdata/config.json --shape lego --out $O/${n}_$T.mp4 > $O/r_${n}_$T.log 2>&1
    tail -1 $O/r_${n}_$T.log; done
  touch $O/.prep_$T; }
i=0; PIDS=()
for T in 0000 0001 0002 0003; do job $T ${G[$(( i % ${#G[@]} ))]} > $O/job_$T.out 2>&1 & PIDS+=($!); i=$((i+1))
  [ ${#G[@]} -lt 4 ] && [ $(( i % ${#G[@]} )) = 0 ] && { wait "${PIDS[@]}"; PIDS=(); }; done
wait "${PIDS[@]}"; cat $O/job_*.out
for T in 0000 0001 0002 0003; do [ -f $O/.prep_$T ] || { touch $O/TOPBOT4_FAIL; rclone copy $O $R -q; kill $SYNC; exit 1; }; done
CUDA_VISIBLE_DEVICES=${G[0]} PYTHONPATH=$WS/lib python -u $O/spstick_topbot4.py --node_h 0.028284 --out $O --mm_dir $O --edges_from $W/lbours/spstick_Fdist/topbot/topbot_L000.npz > $O/topbot4.log 2>&1
grep -q SPTOPBOT4_OK $O/topbot4.log || { tail -5 $O/topbot4.log; touch $O/TOPBOT4_FAIL; rclone copy $O $R -q; kill $SYNC; exit 1; }
for T in 0000 0001 0002 0003; do rclone copyto $O/spstick_topbot_img_$T.mp4 $V/spstick_topbot_img_$T.mp4 -q; done
kill $SYNC; rclone copy $O $R -q && touch $O/TOPBOT4_DONE; rclone copy $O $R -q
echo TOPBOT4_END
