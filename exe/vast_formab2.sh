#!/bin/bash
# 절 CM119 묶음 formab 2 차 (4090 24 GB 에서 GPU 메모리 부족으로 멈춘 판을 80 GB 카드에서 다시; 명령은 R2 formab/<판>/cmd.txt 그대로, 결과 <판>_80) (exe/vast_extsolve.sh 세팅 그대로). onstart 로 받아 tmux 에서 돎.
#   env: R2_ACCESS_KEY R2_SECRET R2_ENDPOINT XS_COMMIT (R2 lbours/wolfcv/formab/code/formab_<커밋>.tar)
#   클러스터 경로 그대로 흉내 (/home/dkta/work, /dev/shm/dkta -> /workspace/shm, python 링크) -- exe/vast_barstick.sh 와 같은 규약.
#   자료: wolf.pt (R2 lbours/fem_ag_data), 정답 궤적 gt_wolf = R2 _vastmirror/topo/pg_wolf_sand + PhysGaussian wolf_config.json 을 exe/pg_h5_to_pt.py 로 변환 (절 AG 준비와 같은 명령),
#   렌더 = pgmodel/wolf_whitebg-trained · i-physgaussian (R2 _vastmirror). 결과는 판마다 R2 lbours/wolfcv/formab/<판> 로 (덤프 포함), 기준 명령 = R2 lbours/wolfcv/extsolve/cvb/cmd.txt.
#   끝 표식 R2 lbours/wolfcv/formab/vast.END -- 로컬에서 보고 destroy.
set -u
H=${XS_COMMIT:?}; STG=formab
R=r2:storage/result/anchorflow; RX=$R/lbours/wolfcv/formab
mkdir -p /workspace/shm /workspace/tmpd /root/xs
LOG=/root/xs/vast80.log
say() { echo "[vast $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $RX/vast80.log -q 2>/dev/null; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] 커밋 $H 단계 $STG $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,clocks.max.sm,memory.total --format=csv | tee -a $LOG; rclone copyto $LOG $RX/vast80.log -q
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python
ln -sf /usr/local/bin/rclone /home/dkta/bin/rclone
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
if [ ! -f /root/xs/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, h5py, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/xs/pip.done
fi
W=/home/dkta/work; MI=$R/_vastmirror; AG=/dev/shm/dkta/ag_data; mkdir -p $AG
rclone copyto $R/lbours/fem_ag_data/wolf.pt $AG/wolf.pt -q &
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q &
rclone copy $MI/pgmodel/wolf_whitebg-trained $W/pgmodel/wolf_whitebg-trained --transfers 16 -q &
rclone copyto $MI/PhysGaussian/config/wolf_config.json $W/PhysGaussian/config/wolf_config.json -q &
rclone copy $MI/topo/pg_wolf_sand $W/topo/pg_wolf_sand --transfers 16 -q &
D=/root/xs/code/$H; mkdir -p $D; rclone cat $RX/code/formab_$H.tar | tar -x -C $D
wait
cd $D || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8
[ -s $AG/gt_wolf/config.json ] || CUDA_VISIBLE_DEVICES=0 python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /root/xs/gtwolf_unused.pt --out_dir $AG/gt_wolf > /root/xs/gt.log 2>&1
say "[자료] wolf.pt $(stat -c %s $AG/wolf.pt 2>&1) gt_wolf $(ls $AG/gt_wolf 2>/dev/null | wc -l) 개 $(tail -1 /root/xs/gt.log | cut -c1-200) pgmodel $(du -sh $W/pgmodel 2>&1 | cut -f1) 코드 $(ls $D/exe/convabl_an.py 2>&1) pip $(ls /root/xs/pip.done 2>&1)"
[ -f /root/xs/pip.done ] && [ -s $AG/wolf.pt ] && [ -s $AG/gt_wolf/config.json ] && [ -f $D/exe/convabl_an.py ] || { say "[세팅 실패]"; echo "세팅 실패 $(date)" | rclone rcat $RX/vast80.END; exit 1; }
say "[준비 끝] 판 시작: ${FA_RUNS:?}"
O=/home/dkta/work/formab; mkdir -p $O
run() { local n=$1 g=$2 m=${1}_80; mkdir -p $O/$m
  rclone cat $RX/$n/cmd.txt | sed -e "s# --out [^ ]*\$# --out $O/$m#" > $O/$m/cmd.txt; rclone copyto $O/$m/cmd.txt $RX/$m/cmd.txt -q
  say "$m 카드 $g 시작 $(grep -o -- '--out [^ ]*' $O/$m/cmd.txt)"
  env CUDA_VISIBLE_DEVICES=$g timeout 10800 bash -c "$(cat $O/$m/cmd.txt)" < /dev/null > $O/$m/log.txt 2>&1
  say "$m rc=$? $(grep -ah '롤아웃\]' $O/$m/log.txt | tail -1 | cut -c1-200) $(grep -a '발산\|rror' $O/$m/log.txt | tail -1 | cut -c1-200)"
  rclone copy $O/$m $RX/$m -q --transfers 8; say "$m R2 올림 (덤프 포함)"; }
NG=$(nvidia-smi -L | wc -l); PIDS=""; gi=0
for grp in ${FA_RUNS//;/ }; do g=$((gi % NG)); gi=$((gi + 1))
  ( for n in ${grp//,/ }; do run $n $g; done ) & PIDS="$PIDS $!"; done
( while sleep 300; do for d in $O/*_80; do [ -f $d/log.txt ] && tail -c 1500 $d/log.txt | rclone rcat $RX/$(basename $d)/log_tail.txt; done; done ) & WP=$!
wait $PIDS; kill $WP 2>/dev/null
RUNS=$(ls -d $O/*_80 | xargs -n1 basename | tr '\n' ' ')
python -u exe/convabl_an.py $O $RUNS > $O/an80.log 2>&1; say "an rc=$? $(cat $O/an80.log | cut -c1-600 | tr '\n' ' ')"
rclone copy $O $RX --include "an80.log" -q; rclone copyto $O/convabl_an.json $RX/convabl_an_80.json -q
rclone copyto $LOG $RX/vast80.log -q
echo "끝 $(date)" | rclone rcat $RX/vast80.END
