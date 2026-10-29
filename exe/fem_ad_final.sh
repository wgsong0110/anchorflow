#!/bin/bash
# 절 AD 최종 판: F 덤프 켠 실행 (단독 측정 -- fem_c_run.sh EX_TIMED=1: GPU0, gpu0.lock flock) + 3DGS 알파블렌딩 렌더 (렌더는 카드 RG -- GPU0 다른 갈래 측정을 건드리지 않게).
# fem_ex_final.sh 와 같은 렌더 (lb_roll_to_h5 -> lb_render, ficus 는 ψ 색 도). 사용: SHAPE=lego|ficus EX_ROOT=.. RG=3 bash exe/fem_ad_final.sh <태그> <fem_ex_sim.py 인자...>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
TG=${1:?태그}; shift 1
cd "$(dirname "$0")/.." || exit 1
SHAPE=${SHAPE:-lego}; RG=${RG:-3}
if [ "$SHAPE" = ficus ]; then GTD=$W/lbours/ficus_v0/gt; CFG=$GTD/config.json; else GTD=$W/lbdata/test_low/L000; CFG=$W/lbdata/config.json; fi
O=$W/lbours/$EX_ROOT/$TG; R=r2:storage/result/anchorflow/lbours/$EX_ROOT/$TG
EX_TIMED=1 bash exe/fem_c_run.sh $TG 0 "$@"
[ -f $O/roll_fem.pt ] || { echo "[오류] $O/roll_fem.pt 없음"; exit 1; }
rclone copy $O $R -q
[ -f $O/h5_fem/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_fem.pt --traj $GTD --out $O/h5_fem > $O/h5_fem.log 2>&1
mkdir -p $O/vj
[ -f $O/vj/fem_$SHAPE.mp4 ] || CUDA_VISIBLE_DEVICES=$RG python -u exe/lb_render.py --sim $O/h5_fem --config $CFG --shape $SHAPE --out $O/vj/fem_$SHAPE.mp4 \
   --frame_label 0 --caption " " < /dev/null > $O/vj/r_fem.log 2>&1
if [ "$SHAPE" = ficus ]; then
  LO=-2.9999966621398926; HI=4.933352713584899; D=$O/psi; mkdir -p $D
  grep -q PSI_COLOR_DONE $D/psi_color.log 2>/dev/null || CUDA_VISIBLE_DEVICES=$RG python -u exe/psi_color.py --gt_h5 $W/topo/pg_ficus_elastic_v0 --dump $O/roll_fem.pt \
      --config $CFG --out $D --eps 1e-3 --q 1 99 --range $LO $HI --cmap turbo --stats_frames 1 40 125 > $D/psi_color.log 2>&1 < /dev/null
  [ -f $D/psi_fem_ficus.mp4 ] || CUDA_VISIBLE_DEVICES=$RG python -u exe/lb_render.py --sim $O/h5_fem --config $CFG --shape ficus \
      --colors $D/col_ov.npy --out $D/psi_fem_ficus.mp4 > $D/r_fem.log 2>&1 < /dev/null
fi
[ -f $O/vj/fem_$SHAPE.mp4 ] && touch $O/vj/RENDER_DONE || tail -3 $O/vj/r_fem.log
rclone copy $O $R -q
echo "[끝] $TG"
