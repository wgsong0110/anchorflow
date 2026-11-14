#!/bin/bash
# 절 CM160: bread 찢어짐 가설 (--mlow 1e-20 에서 틈의 가벼운 노드가 양쪽 조각을 이어 붙임) 측정. CM157 BA_1·BB_1 명령에 --dump 와 진단 감싸개 exe/tear_wrap.py (결과 불변) 만 더해 다시 돌림.
#   env: R2_ACCESS_KEY R2_SECRET R2_ENDPOINT. 코드 = R2 lbours/wolfcv/cm157/code/cm157_5c82795.tar + R2 lbours/wolfcv/cm160/code/over.tar (새 파일). 끝 표식 R2 lbours/wolfcv/cm160/vast.END
set -u
R=r2:storage/result/anchorflow; RX=$R/lbours/wolfcv/cm157; RO=$R/lbours/wolfcv/cm160${XS_TAG:+/$XS_TAG}
mkdir -p /workspace/shm /workspace/tmpd /root/sv
LOG=/root/sv/vast.log
say() { echo "[vast $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $RO/vast.log -q 2>/dev/null; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,clocks.max.sm,memory.total --format=csv | tee -a $LOG; rclone copyto $LOG $RO/vast.log -q
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
  python -c "import torch, cupy, h5py, scipy, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/sv/pip.done
fi
W=/home/dkta/work; AG=/dev/shm/dkta/ag_data; mkdir -p $AG
rclone copyto $R/lbours/fem_ag_data/bread.pt $AG/bread.pt -q &
MI=$R/_vastmirror
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q &
rclone copy $MI/pgmodel/bread-trained $W/pgmodel/bread-trained --transfers 16 -q &
rclone copy $MI/topo/pg_bread_plastic $W/topo/pg_bread_plastic --transfers 16 -q &
rclone copyto $MI/topo/bread_plastic_config.json $W/topo/bread_plastic_config.json -q &
D=/root/sv/code; mkdir -p $D; rclone cat $RX/code/cm157_5c82795.tar | tar -x -C $D; rclone cat $R/lbours/wolfcv/cm160/code/over.tar | tar -x -C $D
for n in BA_1 BB_1; do rclone cat $RX/$n/cmd.txt > /root/sv/$n.cmd; done
wait
cd $D || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8
PY=/home/dkta/.conda/envs/af/bin/python
CUDA_VISIBLE_DEVICES=0 python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_bread_plastic --config $W/topo/bread_plastic_config.json --out_pt /root/sv/gtb_unused.pt --out_dir $AG/gt_bread --out_dir_n 1 > /root/sv/gt.log 2>&1
say "[자료] bread.pt $(stat -c %s $AG/bread.pt 2>&1) gt1 $(ls $AG/gt_bread | wc -l) cmd $(wc -c < /root/sv/BA_1.cmd) $(wc -c < /root/sv/BB_1.cmd) pip $(ls /root/sv/pip.done 2>&1)"
[ -f /root/sv/pip.done ] && [ -s $AG/bread.pt ] && [ -s $AG/gt_bread/config.json ] && [ -d $W/i-physgaussian/gaussian-splatting ] && [ -d $W/pgmodel/bread-trained ] && [ $(wc -c < /root/sv/BA_1.cmd) -gt 100 ] || { say "[세팅 실패]"; echo "세팅 실패 $(date)" | rclone rcat $RO/vast.END; exit 1; }
V=/home/dkta/work/cm160; mkdir -p $V
$PY -u exe/cm160_an.py gt $AG/bread.pt $V 2>&1 | tee -a $V/an_gt.log
FR=$(cat $V/frames.txt); say "기록 프레임 $FR"
rclone copy $V $RO -q
for n in BA_1 BB_1; do
  mkdir -p $V/$n
  sed -e "s# --out /home/dkta/work/cm157/$n# --dump --out $V/$n#" -e "s#AF_SP_FR=1,300#AF_SP_FR=1,1 AF_TEAR_FR=$FR#" -e "s# exe/startpt_wrap.py# exe/tear_wrap.py exe/startpt_wrap.py#" /root/sv/$n.cmd > $V/$n/cmd.txt
  grep -c -- "--dump --out $V/$n" $V/$n/cmd.txt | grep -q 1 && grep -q "exe/tear_wrap.py" $V/$n/cmd.txt && grep -q "AF_TEAR_FR=$FR" $V/$n/cmd.txt || { say "$n 명령 치환 실패"; continue; }
  env CUDA_VISIBLE_DEVICES=0 timeout 3600 bash -c "$(cat $V/$n/cmd.txt)" < /dev/null > $V/$n/log.txt 2>&1
  say "$n rc=$? $(grep -ah '롤아웃\]' $V/$n/log.txt | tail -1 | cut -c1-200) pred $(stat -c %s $V/$n/pred.pt 2>&1) tear $(ls $V/$n/tear 2>/dev/null | wc -l) $(cat $V/$n/tearinfo.json 2>/dev/null | cut -c1-200)"
  rclone copy $V/$n $RO/$n -q --transfers 8
done
say "[시뮬 끝] 분석"
$PY -u exe/cm160_an.py part $AG/bread.pt $V/BA_1 $V/BB_1 $V 2>&1 | tee -a $V/an_part.log
$PY -u exe/cm160_an.py node $V/BA_1 $V/BB_1 $V 2>&1 | tee -a $V/an_node.log
rclone copy $V $RO -q --transfers 8
say "[끝] R2 대조 $(rclone check $V $RO --one-way -q 2>&1 | grep -c ERROR) 개 차이"
echo "끝 $(date)" | rclone rcat $RO/vast.END
