#!/bin/bash
# FEM 실시간 판 (doc/fem_realtime.md) 결과 3DGS 렌더: fem_roll.sh 와 같은 렌더 경로 (lb_roll_to_h5 -> lb_render 알파블렌딩, 프레임 번호·한국어 라벨)
# 정답 | FEM 나란히. 정답 렌더는 기존 lbours/fem2_lego/gt_lego.mp4 재사용 (없으면 새로 렌더).
# 사용: bash exe/fem_rt_render.sh <태그> <GPU> "<라벨>"     출력 lbours/fem_rt_lego/<태그>/fem_lego.mp4, cmp_lego.mp4
export PYTHONUTF8=1 OMP_NUM_THREADS=4
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
TG=${1:?태그}; G=${2:?GPU}; LB=${3:?라벨}
cd "$(dirname "$0")/.." || exit 1
O=$W/lbours/fem_rt_lego/$TG; R=r2:storage/result/anchorflow/lbours/fem_rt_lego/$TG
GTD=$W/lbdata/test_low/L000; CFG=$W/lbdata/config.json; GT=$W/lbours/fem2_lego/gt_lego.mp4
[ -f $O/roll_fem.pt ] || { echo "[오류] $O/roll_fem.pt 없음"; exit 1; }
[ -f $O/h5_fem/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_fem.pt --traj $GTD --out $O/h5_fem > $O/h5_fem.log 2>&1
[ -f $O/fem_lego.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_fem --config $CFG --shape lego --out $O/fem_lego.mp4 \
   --frame_label 0 --caption "$LB" < /dev/null > $O/r_fem.log 2>&1
[ -f $O/fem_lego.mp4 ] || { echo "[오류] 렌더 실패 $(tail -3 $O/r_fem.log)"; exit 1; }
if [ ! -f $GT ]; then
  GT=$O/gt_lego.mp4
  CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $GTD --config $CFG --shape lego --out $GT --frame_label 0 --caption "정답" < /dev/null > $O/r_gt.log 2>&1
fi
python - $GT $O/fem_lego.mp4 $O/cmp_lego.mp4 <<'PY'
import sys, numpy as np, imageio.v2 as iio
ra, rb = iio.get_reader(sys.argv[1]), iio.get_reader(sys.argv[2])
fps = ra.get_meta_data().get("fps", 30)
A, Bv = [f for f in ra], [f for f in rb]
n = max(len(A), len(Bv)); w = iio.get_writer(sys.argv[3], fps=fps, codec="libx264", quality=8, macro_block_size=1)
for i in range(n):
    a, b = A[min(i, len(A) - 1)], Bv[min(i, len(Bv) - 1)]
    hh = max(a.shape[0], b.shape[0])
    pad = lambda f: np.pad(f, ((0, hh - f.shape[0]), (0, 0), (0, 0)), constant_values=255)
    w.append_data(np.concatenate([pad(a), pad(b)], 1))
w.close(); print(f"[저장] {sys.argv[3]} 프레임 {n} (정답 {len(A)}, FEM {len(Bv)})")
PY
rclone copy $O $R -q
[ -f $O/cmp_lego.mp4 ] && touch $O/RENDER_DONE
