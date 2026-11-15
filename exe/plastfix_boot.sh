#!/bin/bash
# 절 CM189 (plastfix): vast 인스턴스 세팅만 (A6000 1 장). 장면 chair_paste·drums_honey 의 정답 (R2 datasets/anchorflow/nshape/gt) ·3DGS 모델·GF 렌더 코드·우리 판 명령 (lbours/nshape/ours/<장면>_K1/cmd.txt) 을
# vid4_boot.sh 와 같은 경로 (/root/gfs) 에 두고, 코드 묶음 PF_CODE (R2 af_code.tar) 를 /root/af 에 푼다. 실행은 exe/plastfix_run.sh.   env: R2_*, PF_CODE
set -u
R=r2:storage/result/anchorflow; MI=$R/_vastmirror; DS=r2:storage/datasets/anchorflow/nshape; MK=$R/lbours/plastfix
mkdir -p /workspace/shm /workspace/tmpd /root/pr; LOG=/root/pr/boot.log
say() { echo "[plastfix $(date '+%F %T')] $*" | tee -a $LOG; rclone copyto $LOG $MK/boot_${CONTAINER_ID:-x}.log -q 2>/dev/null; }
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,memory.total --format=csv | tee -a $LOG
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1); PD=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
if [ -n "$PL" ] && [ -n "$PD" ] && [ "$PL" -lt "$PD" ]; then say "[전력 제한] $PL < $PD"; echo "POWERCAP" > /root/pr/BOOT_FAIL; exit 1; fi
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python; ln -sf /usr/local/bin/rclone /home/dkta/bin/rclone
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
if [ ! -f /root/pr/pip.done ]; then
  pip install -q --default-timeout=120 "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image scikit-learn lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib 2>&1 | tail -1 | tee -a $LOG
  pip install -q --only-binary=:all: PyMCubes 2>&1 | tail -1
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -1
  python -c "import torch, warp, cupy, h5py, diff_gaussian_rasterization, simple_knn; print('[pip] ok', torch.__version__, torch.cuda.is_available())" 2>&1 | tail -1 | tee -a $LOG && touch /root/pr/pip.done
fi
W=/home/dkta/work; G=/root/gfs; AF=/root/af; RD=/root/afr; mkdir -p $G/ag $G/hf/config $G/hf/model $AF $RD
rclone copy $MI/GaussianFluent $G/GaussianFluent --exclude "model/**" --exclude ".git/**" --exclude "demo/**" --exclude "static/**" --transfers 32 -q &
rclone copy $MI/i-physgaussian $W/i-physgaussian --exclude ".git/**" --transfers 16 -q &
rclone copy $MI/PhysGaussian_upstream_8339ed6 $W/PhysGaussian_upstream --transfers 32 -q &
rclone copyto $R/lbours/p2s_vid/top/cam_patch.py /root/cam_patch.py -q &
rclone copyto $R/lbours/fem_cm/code/NotoSansCJKkr-Regular.otf $W/font.otf -q &
for SC in chair_paste drums_honey; do OBJ=${SC%%_*}; OD=$R/lbours/nshape/ours/${SC}_K1; mkdir -p $G/ours/${SC}_K1
  rclone copy $DS/model/$OBJ $G/hf/model/$OBJ --transfers 16 -q &
  rclone copy $DS/gt/$SC/simulation_ply $G/gt/$SC/simulation_ply --transfers 16 -q &
  rclone copyto $OD/cfg_floor.json $G/ours/${SC}_K1/cfg_floor.json -q; rclone copyto $OD/cmd.txt /root/pr/$SC.cmd -q
  rclone copyto $DS/gt/$SC/config.json $G/hf/config/$SC.json -q
done
rclone cat ${PF_CODE:?}/af_code.tar | tar -x -C $AF
rclone cat $R/lbours/nshape/code_pg_7f3e801/af_code.tar | tar -x -C $RD
wait
ln -sfn $W/PhysGaussian_upstream $W/PhysGaussian
for SC in chair_paste drums_honey; do OBJ=${SC%%_*}; mkdir -p $W/pgmodel; ln -sfn $G/hf/model/$OBJ $W/pgmodel/$OBJ; done
python $AF/exe/patch_gf_bg.py $G/GaussianFluent/gs_simulation.py.orig $G/GaussianFluent/gs_orig_run.py 2>&1 | tail -1
cd $AF; export PYTHONPATH=$G/GaussianFluent:$G/GaussianFluent/gaussian-splatting
for SC in chair_paste drums_honey; do
  python -u exe/pg_h5_to_pt.py --h5_dir $G/gt/$SC/simulation_ply --config $G/ours/${SC}_K1/cfg_floor.json --out_pt $G/ag/$SC.pt --out_dir $G/ag/gt_$SC --out_dir_n 1 --allow_nonfinite > /root/pr/cv_$SC.log 2>&1
  python -u exe/pg_h5_to_pt.py --h5_dir $G/gt/$SC/simulation_ply --config $G/ours/${SC}_K1/cfg_floor.json --out_pt /root/pr/gtfull_$SC.pt --out_dir $G/ag/gtfull_$SC --allow_nonfinite > /root/pr/cvfull_$SC.log 2>&1
  say "[변환] $SC $(tail -1 /root/pr/cv_$SC.log | cut -c1-150) | 전체 $(ls $G/ag/gtfull_$SC | grep -c h5) h5"
done
# 렌더 코드 (code_pg_7f3e801): 장면 이름 -> 물체 이름, 위에서 내려다보는 카메라 패치 (vid4 와 같음)
for SC in chair_paste drums_honey; do OBJ=${SC%%_*}
python - $RD/exe/lb_render.py $SC $OBJ <<'P'
import sys; f, sc, ob = sys.argv[1:]; s = open(f).read(); old = 'MODEL.update({"chair_plasticine"'
if '"%s": "%s"' % (sc, ob) not in s:
    assert old in s; s = s.replace(old, 'MODEL.update({"%s": "%s"})\n%s' % (sc, ob, old), 1); open(f, "w").write(s)
print("MODEL +", sc, ob)
P
done
(cd $RD && python /root/cam_patch.py exe/lb_render.py) 2>&1 | tail -1 | tee -a $LOG
touch /root/pr/BOOT_DONE; say "[세팅 끝]"
