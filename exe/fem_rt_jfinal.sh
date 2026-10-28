#!/bin/bash
# doc/fem_realtime.md 절 J 최종 판: F 덤프 켠 80 프레임 (속도 측정 단독) + 3DGS 렌더 (lb_roll_to_h5 -> lb_render 알파블렌딩, 프레임 번호, 설명 글자 비움 --
# 번호·판 이름·ms·fps·오차 띠는 로컬 exe/fem_rt_compose_j.py 가 합성). 사용: bash exe/fem_rt_jfinal.sh <태그> <GPU> <fem_rt_sim.py 인자...>
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
TG=${1:?태그}; G=${2:?GPU}; shift 2
cd "$(dirname "$0")/.." || exit 1
O=$W/lbours/fem_rt_lego/$TG; R=r2:storage/result/anchorflow/lbours/fem_rt_lego/$TG
[ -f $O/RT_DONE ] || bash exe/fem_rt_sim.sh $TG $G "$@"
[ -f $O/roll_fem.pt ] || { echo "[오류] $O/roll_fem.pt 없음"; exit 1; }
rclone copy $O $R -q
GTD=$W/lbdata/test_low/L000; CFG=$W/lbdata/config.json
[ -f $O/h5_fem/meta.json ] || python -u exe/lb_roll_to_h5.py --dump $O/roll_fem.pt --traj $GTD --out $O/h5_fem > $O/h5_fem.log 2>&1
mkdir -p $O/vj
[ -f $O/vj/fem_lego.mp4 ] || CUDA_VISIBLE_DEVICES=$G python -u exe/lb_render.py --sim $O/h5_fem --config $CFG --shape lego --out $O/vj/fem_lego.mp4 \
   --frame_label 0 --caption " " < /dev/null > $O/vj/r_fem.log 2>&1
[ -f $O/vj/fem_lego.mp4 ] && touch $O/vj/RENDER_DONE || tail -3 $O/vj/r_fem.log
rclone copy $O $R -q
