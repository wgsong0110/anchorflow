#!/bin/bash
# [final_fix_phys] vast.ai onstart: 파레토 방법 x 장면 궤적 (우리 = final_fix runs_nd 덤프, 정답 MPM = 정답 궤적, 베이스라인 = 대표 설정 덤프 판 다시)
#   -> 물리 지표 (exe/ffx_phys.py) + 같은 카메라 3DGS 렌더 (exe/lb_roll_to_h5.py + exe/lb_render.py) -> 시각 지표 (exe/ffx_vis.py).
#   코드 = 베이스라인 실행기 판 fpabl_<FB_COMMIT>.tar (브랜치 final_fix_vrgs: VR-GS 충돌 판, PhysTwin·Spring-Gaus 실행기) + 이 판 파일 ffxphys_<FX_COMMIT>.tar 덮어씀.
#   설정은 exe/vast_fpabl_boot.sh 와 같은 순서 (pip·3DGS 휠·stab_bl_setup.sh·장면 자료). 결과 R2 lbours/final_fix/phys/<방법>_<장면>_n<N>/, 표식 lbours/final_fix/marks/final_fix_phys.*
#   env: R2_* FB_COMMIT FX_COMMIT [FX_HOURS 기한, 기본 10] CONTAINER_ID
set -u
R=r2:storage/result/anchorflow; RX=$R/lbours/paper_remeasure2; OX=$R/lbours/final_pa/bl; FX=$R/lbours/final_fix; RP=$FX/phys; MK=$FX/marks/final_fix_phys
mkdir -p /workspace/shm /workspace/tmpd /root/pr
LOG=/root/pr/vast.log
say() { echo "[fxp $(date '+%F %T')] $*" | tee -a $LOG; rclone copyto $LOG $MK.log -q 2>/dev/null; }
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
say "[시작] 코드 $FB_COMMIT + $FX_COMMIT $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,memory.total --format=csv | tee -a $LOG; free -g | tee -a $LOG
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1); PD=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
if [ -n "$PL" ] && [ -n "$PD" ] && [ "$PL" -lt "$PD" ]; then echo "전력 제한 $PL < $PD $(date)" | rclone rcat $MK.POWERCAP; say "[전력 제한] 일 안 함"; exit 1; fi
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
ln -sf /usr/local/bin/rclone /home/dkta/bin/rclone
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
W=/home/dkta/work; SD=/dev/shm/dkta/anerr_data; AG=/dev/shm/dkta/ag_data; NW=$W/newsc; MI=$R/_vastmirror; mkdir -p $SD $AG
# 자료 내려받기 (pip 와 동시)
rclone copyto $R/vastpkg/L000.pt $SD/L000.pt -q &
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q &
rclone copy $MI/pgmodel $W/pgmodel --transfers 16 -q --include "lego_whitebg-trained/**" --include "bread-trained/**" --include "wolf_whitebg-trained/**" &
rclone copy $MI/lbdata $W/lbdata --transfers 16 -q &
rclone copy $MI/stab/lego/init $W/stab/lego/init -q &
rclone copyto $R/lbours/fem_ag_data/bread.pt $AG/bread.pt -q & rclone copy $MI/topo/pg_bread_plastic $W/topo/pg_bread_plastic --transfers 16 -q &
rclone copyto $MI/topo/bread_plastic_config.json $W/topo/bread_plastic_config.json -q &
rclone copyto $R/lbours/fem_ag_data/wolf.pt $AG/wolf.pt -q & rclone copy $MI/topo/pg_wolf_sand $W/topo/pg_wolf_sand --transfers 16 -q &
rclone copyto $MI/PhysGaussian/config/wolf_config.json $W/PhysGaussian/config/wolf_config.json -q &
D=/root/pr/code; mkdir -p $D; rclone cat $OX/code/fpabl_$FB_COMMIT.tar | tar -x -C $D; rclone cat $FX/code/ffxphys_$FX_COMMIT.tar | tar -x -C $D
if [ ! -f /root/pr/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib vastai 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, warp, cupy, h5py, lpips, diff_gaussian_rasterization, simple_knn; print('[pip] torch', torch.__version__, torch.cuda.is_available())" 2>&1 | tail -1 | tee -a $LOG && touch /root/pr/pip.done
fi
wait
# 시각 지표 pyiqa (인스턴스 안에서만 설치; torch·numpy 등 기존 판 고정)
pip install -q pyiqa -c <(pip freeze | grep -iE "^(torch|torchvision|numpy|scipy|scikit-image|opencv-python-headless)==") 2>&1 | tail -1 | tee -a $LOG
python -c "import pyiqa, torch; print('[pyiqa]', pyiqa.__version__, torch.__version__)" 2>&1 | tail -1 | tee -a $LOG
cd $D || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_pr XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext CUDA_VISIBLE_DEVICES=0
mkdir -p $CUPY_CACHE_DIR
[ -s $AG/gt_bread/config.json ] || python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json --out_pt /root/pr/gtbread_unused.pt --out_dir $AG/gt_bread --out_dir_n 1 > /root/pr/gt_bread.log 2>&1
[ -s $AG/gt_wolf/config.json ] || python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /root/pr/gtwolf_unused.pt --out_dir $AG/gt_wolf > /root/pr/gt_wolf.log 2>&1
PR_SC=pillow2sofa RX=$RX; source exe/pr2_scene_fetch.sh
O=/dev/shm/dkta/lbours/paper_remeasure2/stab; mkdir -p $O/p; ln -sfn $O /dev/shm/dkta/lbours/stab
say "[자료] L000 $(stat -c %s $SD/L000.pt) bread $(stat -c %s $AG/bread.pt) wolf $(stat -c %s $AG/wolf.pt) p2s $(stat -c %s $NW/pillow2sofa.pt) gt $(ls $AG | tr '\n' ' ') pgmodel $(ls $W/pgmodel | tr '\n' ' ')"
echo "[ok] $(date)" | rclone rcat $MK.setup.done
# 베이스라인 실행기 환경 (pip 뒤 따로: 우리·정답 쪽 지표·렌더는 바로 시작)
(
  rclone copy $RX/stab_A/p/phystwin_L000_prep $O/p/phystwin_L000_prep -q --transfers 8
  rclone copy $RX/stab_B/p/springgaus_L000_prep $O/p/springgaus_L000_prep -q --transfers 8
  rclone copy $OX/T1/p/phystwin_pillow2sofa_prep $O/p/phystwin_pillow2sofa_prep -q --transfers 8
  (apt-get install -y -qq libegl1 libgl1 libglib2.0-0 libgomp1 >/dev/null 2>&1); pip install -q cloudpickle dataclasses_json plotly narwhals nbformat ipywidgets dash pyquaternion 2>&1 | tail -1
  rclone copyto $R/vastpkg/pytorch3d-0.7.8-cp311-cp311-linux_x86_64.whl /root/pr/pytorch3d-0.7.8-cp311-cp311-linux_x86_64.whl -q
  ST_P3D_WHL=/root/pr/pytorch3d-0.7.8-cp311-cp311-linux_x86_64.whl ST_NO_BUILD=1 ST_SIMP_PY=/home/dkta/.conda/envs/af/bin/python bash exe/stab_bl_setup.sh /root/pr/bl_env.sh > /root/pr/bl_setup.log 2>&1
  say "[bl_setup] $(tail -1 /root/pr/bl_setup.log)"; rclone copyto /root/pr/bl_setup.log $MK.bl_setup.log -q
  touch /root/pr/bl.done
) &
( while sleep 300; do rclone copy /workspace/fxp $RP -q --transfers 8 --exclude "*/h5/**" --exclude "ours_*/pred.pt"; rclone copyto $LOG $MK.log -q; done ) & SY=$!
bash exe/ffx_phys_drv.sh > /root/pr/drv.log 2>&1
say "[작업 끝] $(tail -2 /root/pr/drv.log | tr '\n' ' ')"
kill $SY 2>/dev/null
rclone copy /workspace/fxp $RP -q --transfers 8 --exclude "*/h5/**" --exclude "ours_*/pred.pt"
rclone copyto /root/pr/drv.log $MK.drv.log -q
if rclone check /workspace/fxp $RP --one-way --size-only -q --exclude "*/h5/**" --exclude "ours_*/pred.pt" 2>/root/pr/check.txt; then
  say "[R2 확인 차이 0]"; echo "$(date)" | rclone rcat $MK.ALL_DONE
else
  say "[R2 확인 차이 있음]"; echo "$(date)" | rclone rcat $MK.ALL_DONE_CHECK_FAIL
fi
