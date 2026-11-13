#!/bin/bash
# wolf 탄성 에너지 색 비교 (정답 | H9 nb9 | 뉴턴 60 ncvb, 절 CM119 판) 의 vast.ai 판 -- exe/vast_extsolve.sh 와 같은 세팅, 일은 psi + 3DGS 색 렌더 셋만.
#   env: R2_ACCESS_KEY R2_SECRET R2_ENDPOINT CV_COMMIT (코드 R2 lbours/wolfcv/cvben/code/cvben_<커밋>.tar = git archive exe lib)
#   색 범위 = 앞 영상 fem_wolf_h9_energy.mp4 (R2 lbours/wolfen/en.json) 의 lo·hi 그대로 고정. 결과 R2 lbours/wolfcv/cvben/{psi,vid}, 끝 표식 cvben/vast.END.
set -u
H=${CV_COMMIT:?}
R=r2:storage/result/anchorflow; RX=$R/lbours/wolfcv/cvben; RS=$R/lbours/wolfcv/extsolve
mkdir -p /workspace/shm /workspace/tmpd /root/cv
LOG=/root/cv/vast.log
say() { echo "[vast $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $RX/vast.log -q 2>/dev/null; }
say "[시작] 커밋 $H $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,clocks.max.sm,memory.total --format=csv | tee -a $LOG; rclone copyto $LOG $RX/vast.log -q
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
if [ ! -f /root/cv/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, h5py, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available())" 2>&1 | tail -1 | tee -a $LOG && touch /root/cv/pip.done
fi
W=/home/dkta/work; MI=$R/_vastmirror; AG=/dev/shm/dkta/ag_data; DD=/workspace/cv; mkdir -p $AG $DD/nb9 $DD/ncvb
rclone copyto $R/lbours/fem_ag_data/wolf.pt $AG/wolf.pt -q &
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q &
rclone copy $MI/pgmodel/wolf_whitebg-trained $W/pgmodel/wolf_whitebg-trained --transfers 16 -q &
rclone copyto $MI/PhysGaussian/config/wolf_config.json $W/PhysGaussian/config/wolf_config.json -q &
rclone copy $MI/topo/pg_wolf_sand $W/topo/pg_wolf_sand --transfers 16 -q &
rclone copyto $RS/nb9/pred.pt $DD/nb9/pred.pt -q &
rclone copyto $RS/ncvb/pred.pt $DD/ncvb/pred.pt -q &
D=/root/cv/code/$H; mkdir -p $D; rclone cat $RX/code/cvben_$H.tar | tar -x -C $D
wait
cd $D || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8
[ -s $AG/gt_wolf/config.json ] || CUDA_VISIBLE_DEVICES=0 python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /root/cv/gtwolf_unused.pt --out_dir $AG/gt_wolf > /root/cv/gt.log 2>&1
GTW=$AG/gt_wolf
say "[자료] wolf.pt $(stat -c %s $AG/wolf.pt 2>&1) gt_wolf $(ls $GTW 2>/dev/null | wc -l) 개 $(tail -1 /root/cv/gt.log 2>/dev/null | cut -c1-200) pred $(du -sh $DD/nb9/pred.pt $DD/ncvb/pred.pt 2>&1 | cut -f1 | tr '\n' ' ') pip $(ls /root/cv/pip.done 2>&1)"
[ -f /root/cv/pip.done ] && [ -s $AG/wolf.pt ] && [ -s $GTW/config.json ] && [ -s $DD/nb9/pred.pt ] && [ -s $DD/ncvb/pred.pt ] || { say "[세팅 실패]"; echo "세팅 실패 $(date)" | rclone rcat $RX/vast.END; exit 1; }
RNG=$(rclone cat $R/lbours/wolfen/en.json | python -c "import json,sys; d=json.load(sys.stdin); print(d['lo'], d['hi'])")
O=$DD/out; mkdir -p $O/psi $O/vid
say "[psi] 색 범위 (앞 영상 그대로) $RNG"
python -u exe/cvben_psi.py $GTW $AG/wolf.pt $O/psi $RNG nb9=$DD/nb9/pred.pt ncvb=$DD/ncvb/pred.pt > $O/psi.log 2>&1
rclone copyto $O/psi.log $RX/psi.log -q; rclone copyto $O/psi/en.json $RX/en.json -q
grep -q CVBEN_PSI_DONE $O/psi.log || { say "[psi 실패]"; echo "psi 실패 $(date)" | rclone rcat $RX/vast.END; exit 1; }
for nm in nb9 ncvb; do python -u exe/lb_roll_to_h5.py --dump $DD/$nm/pred.pt --traj $GTW --out $DD/h5_$nm > $O/h5_$nm.log 2>&1; say "[h5] $nm rc=$? $(tail -1 $O/h5_$nm.log)"; done
RC="exe/lb_render.py --config $GTW/config.json --shape wolf"
python -u $RC --sim $GTW --frames 50 --colors $O/psi/col_gt.npy --out $O/vid/gt_en.mp4 < /dev/null > $O/vid/r_gt.log 2>&1; say "[렌더] gt rc=$?"
for nm in nb9 ncvb; do python -u $RC --sim $DD/h5_$nm --frames 50 --colors $O/psi/col_$nm.npy --out $O/vid/${nm}_en.mp4 < /dev/null > $O/vid/r_$nm.log 2>&1; say "[렌더] $nm rc=$?"; done
rclone copy $O/vid $RX/vid -q; rclone copy $O/psi $RX/psi -q --include "en.json" --include "psi_*.npy"
say "[끝] R2 대조: vid $(rclone check $O/vid $RX/vid --one-way -q 2>&1 | grep -c ERROR) · psi $(rclone check $O/psi $RX/psi --one-way --include 'en.json' --include 'psi_*.npy' -q 2>&1 | grep -c ERROR) 개 차이"
echo "끝 $(date)" | rclone rcat $RX/vast.END
