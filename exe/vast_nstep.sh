#!/bin/bash
# 절 CM9.3 묶음 nstep (vast.ai A6000 1 장 인스턴스 onstart -> tmux nstep; env R2_ACCESS_KEY R2_SECRET R2_ENDPOINT AF_COMMIT, vast 가 주는 CONTAINER_ID·CONTAINER_API_KEY):
#   수박 cm9 w1a 명령 그대로 (exe/fem_cm9_scn.sh wm + --gq_mass --gq_chk --dump) + 진단 --nan_step (결과 불변) -- 비결정 판이라 발산 프레임은 그 실행에서 생긴 대로.
#   발산 안 하면 (100 프레임 완주) 같은 명령을 다시 (최대 3 번, n1·n2·n3). 발산한 첫 판에서 멈춤.
# 출력 R2 lbours/fem_cm/cm9n/<판> (log·summary·fem.jsonl·nan_step_f*.npz·pred.pt), 로그 marks/nstep.log, 끝 표식 marks/nstep.END.
# 끝나면 rclone check 차이 0 일 때 스스로 destroy (-y), 아니면 그대로 두고 marks/nstep_CHECK_FAIL.
set -u
C=${AF_COMMIT:?}
R=r2:storage/result/anchorflow; MR=$R/lbours/fem_cm/marks
mkdir -p /workspace/shm /workspace/tmpd /root/ns
LOG=/root/ns/nstep.log
say() { echo "[nstep $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MR/nstep.log -q 2>/dev/null; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] 커밋 $C $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.draw,clocks.max.sm,memory.total --format=csv | tee -a $LOG
# ---- 클러스터 경로 그대로 (exe/vast_s6.sh 와 같은 규약)
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
PYS=$(command -v python); ln -sf $PYS /home/dkta/.conda/envs/af/bin/python
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
if [ ! -f /root/ns/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib vastai 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, h5py, matplotlib, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/ns/pip.done
fi
[ -f /root/ns/pip.done ] || { say "[세팅 실패] pip"; echo "세팅 실패 $(date)" | rclone rcat $MR/nstep.END; exit 1; }
# ---- 코드 (github, 인스턴스에서 편집 안 함)
[ -d /root/af/.git ] || git clone -q https://github.com/wgsong0110/anchorflow.git /root/af
git -C /root/af fetch -q origin implicit-bspline; git -C /root/af checkout -q $C || { say "[코드 실패] $C"; exit 1; }
D=/dev/shm/dkta/impwt/$C
[ -f $D/.done ] || { rm -rf $D; git clone -q /root/af $D && git -C $D checkout -q $C && echo $C > $D/.done; }
cd $D || exit 1
say "[코드] $(git log --oneline -1)"
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_nstep XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
PY=/home/dkta/.conda/envs/af/bin/python
B=/dev/shm/dkta/lbours/fem_cm/cm9n; RR=$R/lbours/fem_cm/cm9n; M=/dev/shm/dkta/lbours/fem_cm/marks
mkdir -p $B $M $CUPY_CACHE_DIR
export CM_SCN=exe/fem_cm9_scn.sh
set +u
source exe/fem_cm9_scn.sh
cm_data wm; say "[자료] wm.pt $(ls -la $WMD/wm.pt 2>&1 | awk '{print $5}')"
[ -s $WMD/wm.pt ] || { say "[자료 실패]"; echo "자료 실패 $(date)" | rclone rcat $MR/nstep.END; exit 1; }
for V in n1 n2 n3; do
  O=$B/$V; mkdir -p $O
  cm_scene wm --gq_mass --gq_chk --dump --nan_step
  echo "카드 0 $(cat .done) $(date '+%F %T') $SIM $DA $XA" > $O/run_info.txt
  CUDA_VISIBLE_DEVICES=0 timeout 3600 $PY -u $SIM --out $O --tag wm $DA $XA < /dev/null > $O/log.txt 2>&1
  echo "rc=$? $(date '+%F %T')" >> $O/run_info.txt
  say "  $V $(tail -1 $O/run_info.txt) $(grep -a '롤아웃\|rror\|발산\|CM9.3 덤프' $O/log.txt | tail -4 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O $RR/$V -q --transfers 8
  DIV=$($PY -c "import json; print(json.load(open('$O/summary.json')).get('diverged') or 0)" 2>/dev/null || echo err)
  say "  $V 발산 프레임 $DIV"
  [ "$DIV" != 0 ] && break
done
echo "[nstep 끝] $(date '+%F %T')" > $M/nstep.END; rclone copy $M/nstep.END $MR -q
say "[nstep 끝]"
rclone copy $B $RR -q --transfers 8
if rclone check $B $RR --one-way --size-only -q 2>/root/ns/check.txt; then
  say "[R2 확인 차이 0] -> destroy"
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함: $(tail -3 /root/ns/check.txt | tr '\n' ' ')"; echo "$(date) 차이 있음" | rclone rcat $MR/nstep_CHECK_FAIL
fi
