#!/bin/bash
# wolf 비교 영상 s5 재료: cm158 g100 h5 (정답 MPM 격자 0.02) + cm140 TWS+sv (S_구속 + sv, 50 프레임 완주 판) 렌더 (h9offvid 와 같은 방식; exe/vast_k5vid.sh 와 같은 방식). env: R2_ACCESS_KEY R2_SECRET R2_ENDPOINT XS_COMMIT (cm139 코드 tar 커밋)
#   env: R2_ACCESS_KEY R2_SECRET R2_ENDPOINT XS_COMMIT (R2 lbours/wolfcv/cm131/code/cm131_<커밋>.tar), XS_STG (기본 chk,run,an)
#   클러스터 경로 그대로 흉내 (/home/dkta/work, /dev/shm/dkta -> /workspace/shm, python 링크) -- exe/vast_barstick.sh 와 같은 규약.
#   자료: wolf.pt (R2 lbours/fem_ag_data), 정답 궤적 gt_wolf = R2 _vastmirror/topo/pg_wolf_sand + PhysGaussian wolf_config.json 을 exe/pg_h5_to_pt.py 로 변환 (절 AG 준비와 같은 명령),
#   렌더 = pgmodel/wolf_whitebg-trained · i-physgaussian (R2 _vastmirror). 결과는 go_cm131.sh 가 판마다 R2 lbours/wolfcv/cm131/<판> 로 (덤프 포함).
#   렌더 자료 (pgmodel·i-physgaussian) 가 없으면 세팅 실패로 멈춤 (절 CM130 첫 띄움 실패 몫).
#   끝 표식 R2 lbours/wolfcv/cm131/vast.END -- 로컬에서 보고 destroy.
set -u
H=${XS_COMMIT:?}; STG=${XS_STG:-chk,run,an}
R=r2:storage/result/anchorflow; RX=$R/lbours/wolfcv/s5cmp
mkdir -p /workspace/shm /workspace/tmpd /root/sv
LOG=/root/sv/vast.log
say() { echo "[vast $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $RX/vast.log -q 2>/dev/null; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] 커밋 $H 단계 $STG $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
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
W=/home/dkta/work; MI=$R/_vastmirror; AG=/dev/shm/dkta/ag_data; mkdir -p $AG
rclone copyto $R/lbours/fem_ag_data/wolf.pt $AG/wolf.pt -q &
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q &
rclone copy $MI/pgmodel/wolf_whitebg-trained $W/pgmodel/wolf_whitebg-trained --transfers 16 -q &
rclone copyto $MI/PhysGaussian/config/wolf_config.json $W/PhysGaussian/config/wolf_config.json -q &
rclone copy $MI/topo/pg_wolf_sand $W/topo/pg_wolf_sand --transfers 16 -q &
rclone copy $R/lbours/wolfcv/cm158/g100/simulation_ply $W/g100/simulation_ply --transfers 16 -q &
D=/root/sv/code/$H; mkdir -p $D; rclone cat $R/lbours/wolfcv/cm139/code/cm139_$H.tar | tar -x -C $D
wait
cd $D || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8
[ -s $AG/gt_wolf/config.json ] || CUDA_VISIBLE_DEVICES=0 python -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /root/sv/gtwolf_unused.pt --out_dir $AG/gt_wolf > /root/sv/gt.log 2>&1
say "[자료] wolf.pt $(stat -c %s $AG/wolf.pt 2>&1) gt_wolf $(ls $AG/gt_wolf 2>/dev/null | wc -l) 개 pgmodel $(du -sh $W/pgmodel 2>&1 | cut -f1)"
[ -f /root/sv/pip.done ] && [ -s $AG/gt_wolf/config.json ] && [ -n "$(ls $W/pgmodel/wolf_whitebg-trained 2>/dev/null)" ] && [ -d $W/i-physgaussian/gaussian-splatting ] || { say "[세팅 실패]"; echo "세팅 실패 $(date)" | rclone rcat $RX.END; exit 1; }
GTW=$AG/gt_wolf; PY=/home/dkta/.conda/envs/af/bin/python; V=$W/s5; mkdir -p $V
RC="exe/lb_render.py --config $GTW/config.json --shape wolf"
# 1) g100 h5 -> 궤적 디렉토리 (정답과 같은 pg_h5_to_pt) -> 같은 렌더
CUDA_VISIBLE_DEVICES=0 $PY -u exe/pg_h5_to_pt.py --h5_dir $W/g100 --config $W/PhysGaussian/config/wolf_config.json --out_pt /root/sv/g100_unused.pt --out_dir $V/g100d > $V/g100.h5.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY -u $RC --sim $V/g100d --out $V/g100.mp4 --frame_label 0 --caption " " < /dev/null > $V/g100.r.log 2>&1
say "g100 영상 $(stat -c %s $V/g100.mp4 2>&1) $(tail -1 $V/g100.r.log | cut -c1-160)"
# 2) TWS+sv
rclone copyto $R/lbours/wolfcv/cm140/TWS+sv/pred.pt $V/sv.pred.pt -q
python -c "
import torch
D=torch.load('$V/sv.pred.pt',map_location='cpu',weights_only=False,mmap=True); print('F' in D and D['F'] is not None, D['pred'].shape, list(D.keys()))
" 2>&1 | tee $V/fcheck.log
grep -q "^True" $V/fcheck.log || { say "[F 없음]"; rclone copy $V $RX -q --include "*.log"; echo "F 없음 $(date)" | rclone rcat $RX.END; exit 0; }
$PY -u exe/lb_roll_to_h5.py --dump $V/sv.pred.pt --traj $GTW --out $V/h5_sv > $V/sv.h5.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY -u $RC --sim $V/h5_sv --out $V/sv.mp4 --frame_label 0 --caption " " < /dev/null > $V/sv.r.log 2>&1
say "sv 영상 $(stat -c %s $V/sv.mp4 2>&1) $(tail -1 $V/sv.r.log | cut -c1-160)"
rclone copy $V $RX -q --include "*.mp4" --include "*.log"
echo "끝 $(date)" | rclone rcat $RX.END
