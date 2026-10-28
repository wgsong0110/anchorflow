#!/bin/bash
# 절 L 최종 판: F 덤프 켠 실행 (속도 측정 단독) + 3DGS 렌더 (lb_roll_to_h5 -> lb_render 알파블렌딩, 프레임 번호, 설명 글자 비움 --
# 판 이름·ms·fps·오차 띠는 로컬 exe/fem_ex_compose.py 가 합성). ficus 면 ψ 색 (psi_color, 기존 판과 같은 고정 범위) 도.
# 사용: SHAPE=lego|ficus [EX_ROOT=..] bash exe/fem_ex_final.sh <태그> <GPU> <fem_ex_sim.py 인자...>   (EX_ROOT 없으면 fem_ex_<SHAPE>)
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
TG=${1:?태그}; G=${2:?GPU}; shift 2
cd "$(dirname "$0")/.." || exit 1
SHAPE=${SHAPE:-lego}
if [ "$SHAPE" = ficus ]; then export EX_ROOT=${EX_ROOT:-fem_ex_ficus}; GTD=$W/lbours/ficus_v0/gt; CFG=$GTD/config.json; else export EX_ROOT=${EX_ROOT:-fem_ex_lego}; GTD=$W/lbdata/test_low/L000; CFG=$W/lbdata/config.json; fi
O=$W/lbours/$EX_ROOT/$TG; R=r2:storage/result/anchorflow/lbours/$EX_ROOT/$TG
bash exe/fem_ex_run.sh $TG $G "$@"
[ -f $O/roll_fem.pt ] || { echo "[오류] $O/roll_fem.pt 없음"; exit 1; }
rclone copy $O $R -q
[ -f $O/h5_fem/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_fem.pt --traj $GTD --out $O/h5_fem > $O/h5_fem.log 2>&1
mkdir -p $O/vj
[ -f $O/vj/fem_$SHAPE.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_fem --config $CFG --shape $SHAPE --out $O/vj/fem_$SHAPE.mp4 \
   --frame_label 0 --caption " " < /dev/null > $O/vj/r_fem.log 2>&1
if [ "$SHAPE" = ficus ]; then
  LO=-2.9999966621398926; HI=4.933352713584899; D=$O/psi; mkdir -p $D
  grep -q PSI_COLOR_DONE $D/psi_color.log 2>/dev/null || CUDA_VISIBLE_DEVICES=$G python -u exe/psi_color.py --gt_h5 $W/topo/pg_ficus_elastic_v0 --dump $O/roll_fem.pt \
      --config $CFG --out $D --eps 1e-3 --q 1 99 --range $LO $HI --cmap turbo --stats_frames 1 40 125 > $D/psi_color.log 2>&1 < /dev/null
  [ -f $D/psi_fem_ficus.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_fem --config $CFG --shape ficus \
      --colors $D/col_ov.npy --out $D/psi_fem_ficus.mp4 > $D/r_fem.log 2>&1 < /dev/null
fi
[ -f $O/vj/fem_$SHAPE.mp4 ] && touch $O/vj/RENDER_DONE || tail -3 $O/vj/r_fem.log
rclone copy $O $R -q
echo "[끝] $TG"
