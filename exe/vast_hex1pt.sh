#!/bin/bash
# 절 CM179 (육면체 셀 가우스 1 점 대 2x2x2) 의 vast.ai 판 (exe/vast_cm157.sh 그대로, 경로·진입점만). onstart 로 받아 tmux sv 에서 돎.
#   env: R2_ACCESS_KEY R2_SECRET R2_ENDPOINT XS_COMMIT (R2 lbours/hex1pt/code/hex1pt_<커밋>.tar); XS_G0·XS_G1 카드 0·1 판 목록
#   끝 표식 R2 lbours/hex1pt/vast.END -- 로컬에서 보고 destroy.
set -u
H=${XS_COMMIT:?}
R=r2:storage/result/anchorflow; RX=$R/lbours/hex1pt
mkdir -p /workspace/shm /workspace/tmpd /root/sv
LOG=/root/sv/vast.log
say() { echo "[vast $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $RX/vast.log -q 2>/dev/null; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] 커밋 $H $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,clocks.max.sm,memory.total --format=csv | tee -a $LOG; rclone copyto $LOG $RX/vast.log -q
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
rclone copyto $R/vastpkg/L000.pt $SD/L000.pt -q &
rclone copyto $R/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt $SD/F_pg_v0.pt -q &
rclone copyto $R/lbours/fem_ag_data/bread.pt $AG/bread.pt -q &
MI=$R/_vastmirror
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q &
for m in lego_whitebg-trained ficus_whitebg-trained bread-trained; do rclone copy $MI/pgmodel/$m $W/pgmodel/$m --transfers 16 -q & done
rclone copy $MI/lbdata/test_low/L000 $W/lbdata/test_low/L000 -q --include meta.json --include sim_0000000000.h5 &
rclone copyto $MI/lbdata/config.json $W/lbdata/config.json -q &
rclone copyto $MI/stab/lego/init/sim_0000000000.h5 $W/stab/lego/init/sim_0000000000.h5 -q &
rclone copy $R/lbours/ficus_v0/gt $W/lbours/ficus_v0/gt -q --include meta.json --include config.json --include sim_0000000000.h5 &
rclone copy $MI/topo/pg_ficus_elastic_v0 $W/topo/pg_ficus_elastic_v0 -q &
rclone copy $MI/topo/pg_bread_plastic $W/topo/pg_bread_plastic --transfers 16 -q &
rclone copyto $MI/topo/bread_plastic_config.json $W/topo/bread_plastic_config.json -q &
D=/root/sv/code/$H; mkdir -p $D; rclone cat $RX/code/hex1pt_$H.tar | tar -x -C $D
wait
cd $D || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8
[ -s $AG/gt_bread/config.json ] || CUDA_VISIBLE_DEVICES=0 python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json --out_pt /root/sv/gtbread_unused.pt --out_dir $AG/gt_bread --out_dir_n 1 > /root/sv/gt.log 2>&1
say "[자료] L000 $(stat -c %s $SD/L000.pt 2>&1) ficus $(stat -c %s $SD/F_pg_v0.pt 2>&1) bread $(stat -c %s $AG/bread.pt 2>&1) lego init $(ls $W/stab/lego/init 2>&1) ficus topo $(ls $W/topo/pg_ficus_elastic_v0/simulation_ply 2>&1 | head -2 | tr '\n' ' ') gt_bread $(ls $AG/gt_bread | tr '\n' ' ') 코드 $(ls $D/exe/go_hex1pt.sh 2>&1) pip $(ls /root/sv/pip.done 2>&1)"
[ -f /root/sv/pip.done ] && [ -s $SD/L000.pt ] && [ -s $SD/F_pg_v0.pt ] && [ -s $AG/bread.pt ] && [ -f $D/exe/go_hex1pt.sh ] && [ -s $AG/gt_bread/config.json ] && [ -s $W/lbdata/config.json ] && [ -s $W/stab/lego/init/sim_0000000000.h5 ] && [ -s $W/topo/pg_ficus_elastic_v0/simulation_ply/sim_0000000000.h5 ] && [ -d $W/i-physgaussian/gaussian-splatting ] && [ -d $W/pgmodel/bread-trained ] && [ -d $W/pgmodel/lego_whitebg-trained ] && [ -d $W/pgmodel/ficus_whitebg-trained ] || { say "[세팅 실패]"; echo "세팅 실패 $(date)" | rclone rcat $RX/vast.END; exit 1; }
say "[준비 끝] GPU 일 시작"
bash exe/go_hex1pt.sh $H 0 ${XS_G0:?} < /dev/null >> $LOG 2>&1 &
bash exe/go_hex1pt.sh $H 1 ${XS_G1:?} < /dev/null >> $LOG 2>&1 &
wait
rclone copyto $LOG $RX/vast.log -q
say "[끝] R2 대조: $(rclone check /home/dkta/work/hex1pt $RX --one-way -q 2>&1 | grep -c ERROR) 개 차이"
echo "끝 $(date)" | rclone rcat $RX/vast.END
