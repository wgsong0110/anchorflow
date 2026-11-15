#!/bin/bash
# wolffix (11-15) vast.ai 판: exe/vast_cm184pg.sh (wolf·ficus·bread 자료) + exe/vast_cm157.sh (레고 L000 자료), 진입점 exe/go_wfx.sh. onstart 로 받아 tmux 에서 돎.
#   env: R2_ACCESS_KEY R2_SECRET R2_ENDPOINT XS_COMMIT (R2 lbours/wolffix1115/code/wfx_<커밋>.tar), XS_TAG, XS_SPEC (R2 경로 디렉토리: spec_g<카드>.txt 카드마다)
#   결과 R2 lbours/wolffix1115/<태그>/, 끝 표식 R2 lbours/wolffix1115/<태그>/vast.END
set -u
H=${XS_COMMIT:?}; TAG=${XS_TAG:?}
R=r2:storage/result/anchorflow; RX=$R/lbours/wolffix1115/$TAG
mkdir -p /workspace/shm /workspace/tmpd /root/sv
LOG=/root/sv/vast.log
say() { echo "[vast $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $RX/vast_${CONTAINER_ID:-x}.log -q 2>/dev/null; }
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
say "[시작] 커밋 $H 태그 $TAG $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,clocks.max.sm,memory.total --format=csv | tee -a $LOG
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
ln -sf /usr/local/bin/rclone /home/dkta/bin/rclone
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
if [ ! -f /root/sv/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, h5py, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/sv/pip.done
fi
W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; SD=/dev/shm/dkta/anerr_data; mkdir -p $AG $SD
MI=$R/_vastmirror
rclone copyto $R/lbours/fem_ag_data/wolf.pt $AG/wolf.pt -q &
rclone copyto $MI/PhysGaussian/config/wolf_config.json $W/PhysGaussian/config/wolf_config.json -q &
rclone copy $MI/topo/pg_wolf_sand $W/topo/pg_wolf_sand --transfers 16 -q &
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q &
for m in wolf_whitebg-trained lego_whitebg-trained ficus_whitebg-trained bread-trained; do rclone copy $MI/pgmodel/$m $W/pgmodel/$m --transfers 16 -q & done
rclone copyto $R/vastpkg/L000.pt $SD/L000.pt -q &
rclone copy $MI/lbdata/test_low/L000 $W/lbdata/test_low/L000 -q --include meta.json --include sim_0000000000.h5 &
rclone copyto $MI/lbdata/config.json $W/lbdata/config.json -q &
rclone copyto $MI/stab/lego/init/sim_0000000000.h5 $W/stab/lego/init/sim_0000000000.h5 -q &
rclone copyto $R/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt $SD/F_pg_v0.pt -q &
rclone copy $R/lbours/ficus_v0/gt $W/lbours/ficus_v0/gt -q --include meta.json --include config.json --include sim_0000000000.h5 &
rclone copy $MI/topo/pg_ficus_elastic_v0 $W/topo/pg_ficus_elastic_v0 -q &
rclone copyto $R/lbours/fem_ag_data/bread.pt $AG/bread.pt -q &
rclone copy $MI/topo/pg_bread_plastic $W/topo/pg_bread_plastic --transfers 16 -q &
rclone copyto $MI/topo/bread_plastic_config.json $W/topo/bread_plastic_config.json -q &
D=/root/sv/code/$H; mkdir -p $D; rclone cat $R/lbours/wolffix1115/code/wfx_$H.tar | tar -x -C $D
mkdir -p /root/sv/spec; rclone copy ${XS_SPEC:?} /root/sv/spec -q
wait
cd $D || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8
[ -s $AG/gt_wolf/config.json ] || CUDA_VISIBLE_DEVICES=0 python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /root/sv/gtwolf_unused.pt --out_dir $AG/gt_wolf > /root/sv/gtw.log 2>&1
[ -s $AG/gt_bread/config.json ] || CUDA_VISIBLE_DEVICES=0 python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json --out_pt /root/sv/gtbread_unused.pt --out_dir $AG/gt_bread --out_dir_n 1 > /root/sv/gt.log 2>&1
say "[자료] wolf $(stat -c %s $AG/wolf.pt 2>&1) gt_wolf $(ls $AG/gt_wolf | wc -l) L000 $(stat -c %s $SD/L000.pt 2>&1) ficus $(stat -c %s $SD/F_pg_v0.pt 2>&1) bread $(stat -c %s $AG/bread.pt 2>&1) pgmodel $(ls $W/pgmodel | tr '\n' ' ') spec $(ls /root/sv/spec | tr '\n' ' ')"
[ -f /root/sv/pip.done ] && [ -s $AG/wolf.pt ] && [ -s $AG/gt_wolf/config.json ] && [ -d $W/pgmodel/wolf_whitebg-trained ] && [ -f $D/exe/go_wfx.sh ] && [ -d $W/i-physgaussian/gaussian-splatting ] || { say "[세팅 실패]"; echo "세팅 실패 $(date)" | rclone rcat $RX/vast.END; exit 1; }
say "[준비 끝]"
for f in /root/sv/spec/spec_g*.txt; do
  g=$(basename $f .txt); g=${g#spec_g}
  tmux new-session -d -s wfx$g "cd $D && bash exe/go_wfx.sh $TAG $g $f > /root/sv/go_g$g.out 2>&1"
done
while tmux ls 2>/dev/null | grep -q '^wfx'; do sleep 30; done
say "[끝]"
echo "끝 $(date)" | rclone rcat $RX/vast.END
