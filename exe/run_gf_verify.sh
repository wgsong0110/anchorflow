#!/bin/bash
# GPU 가 잡히면 이것 하나만 돌리면 된다.
#
#   bash exe/run_gf_verify.sh
#
# 1) 씬 스윕   재질 6 x 전달 4 x 경계·구동 8 x 구역물성 = 20 개를 GF 와 이식본으로
#              각각 굴려 입자별로 맞춰 본다. "임의의 씬에서 같다" 는 여기서 본다
# 2) 수박 맞추기  실제 파괴 궤적(watermelon_h)을 같은 초기상태에서 이어 굴리고
#              인열 잣대와 좌우 비교 영상을 낸다
set -u
W=${W:-$HOME/work}
R=${R:-$W/anchorflow}
LOG=$W/gfverify.log
exec > "$LOG" 2>&1
source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af
cd "$R"; git pull -q origin master || true
echo "[노드] $(hostname)  GPU=${CUDA_VISIBLE_DEVICES:-?}  커밋 $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.used,power.limit --format=csv,noheader

# ---------------------------------------------------------------- 1) 씬 스윕
python -u exe/make_gf_scenes.py --out "$W/gfscn" --n_grid 100 --frames 12
python -u exe/verify_gf_port.py --cfg_dir "$W/gfscn" \
  --gf_root "$W/GaussianFluent" --model "$W/pgmodel/bread-trained" \
  --work "$W/gfver" --out "$W/gfver_summary.json" --rm_h5 --tol 0.05

# ------------------------------------------------------- 2) 수박 파괴 맞추기
# ref_wmh 는 GF 의 watermelon_h (n_grid 300, FLIP 0.7, g=[4,0,-15]) 를 전체
# 해상도로 다시 돌린 것. 씬 전용 러너라 --flip on, --auto_dt 가 맞다.
if [ -f "$W/ref_wmh/simulation_ply/sim_0000000000.h5" ]; then
  rm -rf "$W/my_wmh"
  python -u exe/gf_mpm.py --config "$W/cfg_wmh40.json" \
    --h5 "$W/ref_wmh/simulation_ply/sim_0000000000.h5" --out "$W/my_wmh" \
    --auto_dt --flip on --frames 30
  echo '--- 입자별 비교 ---'
  python -u exe/compare_solvers.py --a "$W/ref_wmh" --b "$W/my_wmh" \
    --tag_a GF --tag_b MINE --every 5 --out "$W/cmp_wmh.json"
  echo '--- 인열 (GF) ---'
  python -u exe/measure_gf_tearing.py --h5_dir "$W/ref_wmh" --tag GF --out "$W/tear_ref"
  echo '--- 인열 (이식본) ---'
  python -u exe/measure_gf_tearing.py --h5_dir "$W/my_wmh" --tag MINE --out "$W/tear_my"
  echo '--- 영상 ---'
  python -u exe/render_gf_traj.py --h5_dir "$W/ref_wmh" --tag gf_wmh --out "$W/cdvid" \
    --stride 1 --fps 10 --width 520 --elev 12 --azim 25
  python -u exe/render_gf_traj.py --h5_dir "$W/my_wmh" --tag my_wmh --out "$W/cdvid" \
    --stride 1 --fps 10 --width 520 --elev 12 --azim 25
  ffmpeg -y -loglevel error -i "$W/cdvid/gf_wmh.mp4" -i "$W/cdvid/my_wmh.mp4" \
    -filter_complex "[0:v]drawtext=text='GaussianFluent':x=10:y=10:fontsize=20:fontcolor=white[a];[1:v]drawtext=text='MPM only (ported)':x=10:y=10:fontsize=20:fontcolor=white[b];[a][b]hstack" \
    "$W/cdvid/wm_match.mp4" && echo "[저장] $W/cdvid/wm_match.mp4"
else
  echo "[건너뜀] $W/ref_wmh 가 없다 -- GF 로 watermelon_h 를 먼저 다시 돌려야 한다"
fi
echo GF_VERIFY_DONE
