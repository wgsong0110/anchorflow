#!/bin/bash
# 절 CM126 묶음 escwhy (wolf 입자 이탈 원인 걸음별 기록) 의 vast.ai 판 (exe/vast_formab2.sh 세팅 그대로). onstart 로 받아 tmux 에서 돎.
#   env: R2_ACCESS_KEY R2_SECRET R2_ENDPOINT XS_COMMIT (R2 lbours/wolfcv/escwhy/code/escwhy_<커밋>.tar)
#        ESC_RUNS = 카드별 묶음 '+', 묶음 안 차례 ',', 판 = 이름:원본판:F0[:F1][:추가플래그(=는 공백, ~ 는 구분)]
#          원본판 = R2 lbours/wolfcv/formab/<원본판>/cmd.txt 를 그대로 쓰고 python -u 바로 뒤에 exe/escwhy_wrap.py exe/nlwhy_wrap.py 를 끼움 (기록만, 결과 불변).
#   결과: 판마다 R2 lbours/wolfcv/escwhy/<이름> (cmd·log·fem.jsonl·nl.jsonl·pred.pt·esc/fNNN.npz 모두, 60 초마다 + 끝에 통째로 동기화, 아무것도 지우지 않음).
#   끝 표식 R2 lbours/wolfcv/escwhy/vast_<ESC_TAG>.END -- 로컬에서 보고 destroy.
set -u
H=${XS_COMMIT:?}; TAG=${ESC_TAG:-a}
R=r2:storage/result/anchorflow; RX=$R/lbours/wolfcv/escwhy; RF=$R/lbours/wolfcv/formab
mkdir -p /workspace/shm /workspace/tmpd /root/xs
LOG=/root/xs/vast_$TAG.log
say() { echo "[vast $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $RX/vast_$TAG.log -q 2>/dev/null; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] 커밋 $H 판 ${ESC_RUNS:?} $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,clocks.max.sm,memory.total --format=csv | tee -a $LOG; rclone copyto $LOG $RX/vast_$TAG.log -q
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
D=/root/xs/code/$H; mkdir -p $D; rclone cat $RX/code/escwhy_$H.tar | tar -x -C $D
wait
cd $D || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8
[ -s $AG/gt_wolf/config.json ] || CUDA_VISIBLE_DEVICES=0 python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /root/xs/gtwolf_unused.pt --out_dir $AG/gt_wolf > /root/xs/gt.log 2>&1
say "[자료] wolf.pt $(stat -c %s $AG/wolf.pt 2>&1) gt_wolf $(ls $AG/gt_wolf 2>/dev/null | wc -l) 개 $(tail -1 /root/xs/gt.log | cut -c1-200) 코드 $(ls $D/lib/anchorflow/fem_escwhy.py 2>&1) pip $(ls /root/xs/pip.done 2>&1)"
[ -f /root/xs/pip.done ] && [ -s $AG/wolf.pt ] && [ -s $AG/gt_wolf/config.json ] && [ -f $D/lib/anchorflow/fem_escwhy.py ] || { say "[세팅 실패]"; echo "세팅 실패 $(date)" | rclone rcat $RX/vast_$TAG.END; exit 1; }
O=/home/dkta/work/escwhy; mkdir -p $O
run() { local spec=$1 g=$2; IFS=: read -r n src f0 f1 xf <<< "$spec"; f1=${f1:-1000}; xf=${xf:-}; mkdir -p $O/$n
  rclone cat $RF/$src/cmd.txt | sed -e "s# --out [^ ]*\$# --out $O/$n#" -e "s#/bin/python -u exe/#/bin/python -u exe/escwhy_wrap.py exe/nlwhy_wrap.py exe/#" > $O/$n/cmd.txt
  if [ -n "$xf" ]; then x2=$(echo "$xf" | tr '=~' '  '); sed -i -e "s# --out # $x2 --out #" $O/$n/cmd.txt; fi
  echo "ESC_F0=$f0 ESC_F1=$f1 $(cat $O/$n/cmd.txt)" > $O/$n/cmd.txt
  rclone copyto $O/$n/cmd.txt $RX/$n/cmd.txt -q
  say "$n (원본 $src, 기록 f$f0~$f1, 추가 '$xf') 카드 $g 시작: $(grep -o 'escwhy_wrap.py exe/nlwhy_wrap.py exe/[a-z0-9_]*.py' $O/$n/cmd.txt) $(grep -o -- '--out [^ ]*' $O/$n/cmd.txt)"
  env CUDA_VISIBLE_DEVICES=$g timeout 10800 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  say "$n rc=$? $(grep -ah '롤아웃\]' $O/$n/log.txt | tail -1 | cut -c1-200) $(grep -a '발산\|rror' $O/$n/log.txt | tail -1 | cut -c1-200) 기록 $(ls $O/$n/esc 2>/dev/null | wc -l) 프레임"
  rclone copy $O/$n $RX/$n -q --transfers 8; say "$n R2 올림 (덤프 포함)"; }
( while sleep 60; do rclone copy $O $RX -q --transfers 8; done ) & WP=$!
NG=$(nvidia-smi -L | wc -l); PIDS=""; gi=0
for grp in ${ESC_RUNS//+/ }; do g=$((gi % NG)); gi=$((gi + 1))
  ( for sp in ${grp//,/ }; do run $sp $g; done ) & PIDS="$PIDS $!"; done
wait $PIDS; kill $WP 2>/dev/null
rclone copy $O $RX -q --transfers 8
say "[끝] R2 대조: $(rclone check $O $RX --one-way -q 2>&1 | grep -c ERROR) 개 차이"
rclone copyto $LOG $RX/vast_$TAG.log -q
echo "끝 $(date)" | rclone rcat $RX/vast_$TAG.END
