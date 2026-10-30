#!/bin/bash
# 절 AG 렌더: 씬마다 단독 측정 회차 1 의 F 덤프 -> h5 -> 3DGS 알파블렌딩 (PG 씬 lb_render, 수박 GF 러너 규약 wm_render) + 정답 같은 렌더러.
# AG_TIMED="씬:N ..." AG_CARDS (카드 목록, 씬마다 하나씩 돌려 씀). R2 에 같은 크기로 올라간 것 확인 뒤 /tmp 큰 덤프만 지움.
cd "$(dirname "$0")/.." || exit 1
WT=$(pwd)
export PYTHONUTF8=1 OMP_NUM_THREADS=8; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
source exe/fem_ag_scene.sh
RR=r2:storage/result/anchorflow/lbours; CARDS=(${AG_CARDS:-1})
WMM=${AG_WM_MODEL:-$W/gfmodel/watermelon}
rend() {
  local S=$1 N=$2 G=$3; ag_scene $S
  local O=/tmp/dkta/lbours/fem_ag_$S/ag_${S}_n${N}_t1 GD=/tmp/dkta/lbours/fem_ag_$S/gt
  [ -f $O/roll_fem.pt ] || { echo "[덤프 없음] $S"; return; }
  mkdir -p $O/vj $GD
  [ -f $O/h5_fem/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_fem.pt --traj $AGD/gt_$S --out $O/h5_fem > $O/h5_fem.log 2>&1
  if [ $S = wm ]; then
    [ -f $O/vj/fem_$SHAPE.mp4 ] || (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$G python -u $WT/exe/wm_render.py --model_path $WMM --config $CFG --sim $O/h5_fem --out $O/vj/fem_$SHAPE.mp4 < /dev/null > $O/vj/r_fem.log 2>&1)
    [ -f $GD/gt_$SHAPE.mp4 ] || (cd $W/GaussianFluent && CUDA_VISIBLE_DEVICES=$G python -u $WT/exe/wm_render.py --model_path $WMM --config $CFG --sim $GTH5 --out $GD/gt_$SHAPE.mp4 < /dev/null > $GD/r_gt.log 2>&1)
  else
    [ -f $O/vj/fem_$SHAPE.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_fem --config $AGD/gt_$S/config.json --shape $SHAPE --out $O/vj/fem_$SHAPE.mp4 \
       --frame_label 0 --caption " " ${AG_CAMB:+--cam_back $AG_CAMB} < /dev/null > $O/vj/r_fem.log 2>&1
    [ -f $GD/gt_$SHAPE.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $AGD/gt_$S --config $AGD/gt_$S/config.json --shape $SHAPE --out $GD/gt_$SHAPE.mp4 \
       --frame_label 0 --caption " " ${AG_CAMB:+--cam_back $AG_CAMB} < /dev/null > $GD/r_gt.log 2>&1
  fi
  [ -f $O/vj/fem_$SHAPE.mp4 ] && touch $O/vj/RENDER_DONE || tail -5 $O/vj/r_fem.log
  tail -2 $GD/r_gt.log
  rclone copy $O $RR/fem_ag_$S/ag_${S}_n${N}_t1 -q; rclone copy $GD $RR/fem_ag_$S/gt -q
  if [ -f $O/vj/RENDER_DONE ] && rclone check $O $RR/fem_ag_$S/ag_${S}_n${N}_t1 --one-way --size-only -q 2>/dev/null; then
    rm -rf $O/roll_fem.pt $O/h5_fem; echo "[R2 확인·/tmp 정리] $S"
  else echo "[R2 확인 실패 또는 렌더 실패 -- /tmp 유지] $S"; fi
}
i=0
for SN in ${AG_TIMED:?}; do
  S=${SN%%:*}; N=${SN#*:}; G=${CARDS[$((i % ${#CARDS[@]}))]}
  rend $S $N $G > /tmp/dkta/lbours/fem_ag_sum/rend_$S.log 2>&1 &
  i=$((i + 1))
done
wait
cat /tmp/dkta/lbours/fem_ag_sum/rend_*.log
rclone copy /tmp/dkta/lbours/fem_ag_sum $RR/fem_ag_sum -q
echo "AG_RENDER_DONE $(date +%T)"
