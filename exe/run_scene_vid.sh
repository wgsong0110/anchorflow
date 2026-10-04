#!/bin/bash
# 씬 하나를 **기본 조합**(사면체 + F 장벽)으로 굴려 비교 영상을 뽑는다.
#   bash exe/run_scene_vid.sh wmfrac   # 수박 파괴 (GF CD-MPM 궤적과 나란히)
#   bash exe/run_scene_vid.sh sand     # 모래성 붕괴 (PG sand 궤적과 나란히)
#
# node_h 는 그 씬 MPM dx 의 (2*sqrt2)^(1/3) 배다 (사면체 부피 = dx^3/6).
# 실측 최근접 간격이 dx 와 거의 같아 40k 표본에서도 셀당 입자가 과하지 않다.
set -u
W=/home/dkta/work
S=${1:?wmfrac 또는 sand}
G=${CUDA_VISIBLE_DEVICES:-0}
case "$S" in
  wmfrac) DATA=$W/gftraj;   TAG=watermelon_h; NH=0.00943; LEN=40
          LBL="수박 파괴 (출력만 최적화, 사면체 h=0.00943)"; LEFT="GF CD-MPM" ;;
  sand)   DATA=$W/sandtraj; TAG=wolf_a;       NH=0.01414; LEN=45
          LBL="모래성 붕괴 (출력만 최적화, 사면체 h=0.01414)"; LEFT="PG MPM" ;;
  *) echo "모르는 씬 $S"; exit 1 ;;
esac
U=$(echo "$S" | tr 'a-z' 'A-Z')
export PATH=/home/dkta/.conda/envs/af/bin:$PATH
export AF_WORK=$W PYTHONPATH=$W/anchorflow/lib PYTHONIOENCODING=utf-8
export MPLCONFIGDIR=/home/dkta/.mplcache CUDA_VISIBLE_DEVICES=$G
export PYTHONUTF8=1
cd $W/anchorflow
mkdir -p $W/sceneout
export AF_ROLL_DUMP=$W/sceneout/$U.pt AF_ROLL_TAG=$TAG AF_ROLL_T0=0
python -u exe/train_deform.py --data $DATA --out $W/sceneout --tag $U \
  --no_mat --arch sgnn --node_h $NH --gnn_layers 1 --hidden 128 --obj pts \
  --dt_cond --dt_scale --v_from_dt --det_eps 0.1 --det_w 100 --iters 0 \
  --eval_t0 0 --eval_len $LEN --gpu_data 0 --save_every 100000 \
  --ov_init predict --ov_opt lbfgs --ov_roll 500 \
  --bc_level particle --bc_mode barrier --bc_kappa 1.0 --bc_ext linear \
  --bc_mpm --inv_barrier 1.0 --inv_jhat 0.3 --inv_ext linear \
  < /dev/null 2>&1 | grep -aE "뒤집힘|^\[롤아웃\]|Error|Traceback|out of memory" \
  | tail -6
for PL in xz xy; do
  python -u exe/render_rollout_cmp.py --dump $W/sceneout/$U.pt --plane $PL \
    --color znow --out $W/sceneout/${S}_$PL.mp4 --fps 12 \
    --label "$LBL" --label_left "$LEFT" \
    < /dev/null 2>&1 | grep -a "저장" | tail -1
done
python -u exe/vid_detneg.py --dump $W/sceneout/$U.pt --neg_s 8 --s 0.5 \
  --alpha 0.15 --out $W/sceneout/${S}_detneg.mp4 --label "$LBL (뒤집힌 셀 표시)" \
  < /dev/null 2>&1 | grep -a "저장" | tail -1
echo SCENE_${U}_DONE
