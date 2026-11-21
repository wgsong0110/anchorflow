#!/bin/bash
# 새 카드 (조정자 11-17: 가장 빠른 카드로 옮김) 부팅: 전력 제한 확인 -> 환경 (pip·3DGS 휠·i-physgaussian·pillow2sofa 자료·3DGS 모델) ->
#   exe/sg_train_mv.sh p2s (R2 bl/P2 의 가장 늦은 sg_resume 저장점부터) -> exe/sg_post_p2s.sh P2 (같은 카드 단독 시간 1 회·덤프·지표). 재시작하면 같은 순서 (끝난 단계는 건너뜀).
export HOME=/root
RA=r2:storage/result/anchorflow; RB=$RA/lbours/final_pa/bl; RX=$RA/lbours/paper_remeasure2; MI=$RA/_vastmirror; MK=$RB/P2/marks_mv
L=/root/newcard.log; say() { echo "[newcard $(date -u '+%F %T')] $*" >> $L; rclone copyto $L $MK/newcard.log -q; }
say "부팅 $(hostname) $(nvidia-smi --query-gpu=name,power.limit,power.default_limit,memory.total --format=csv,noheader)"
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1); PD=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
if [ "$PL" -lt "$PD" ]; then say "[전력 제한] $PL < $PD -- 일 안 함"; echo "$PL<$PD" | rclone rcat $MK/NEWCARD_POWERCAP -q; exit 1; fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg libegl1 libgl1 libglib2.0-0 libgomp1 unzip) >/dev/null 2>&1
mkdir -p /workspace/shm /home/dkta/.conda/envs/af/bin /home/dkta/bin /home/dkta/work/newsc /home/dkta/work/pgmodel
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
ln -sf $(command -v python) /home/dkta/.conda/envs/af/bin/python; ln -sf $(command -v rclone) /home/dkta/bin/rclone
export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
W=/home/dkta/work; NW=$W/newsc
( [ -s $NW/pillow2sofa.pt ] || rclone copyto $RX/scenes/pillow2sofa/pillow2sofa.pt $NW/pillow2sofa.pt -q --multi-thread-streams 16 ) &
( [ -d $NW/gt_pillow2sofa ] || rclone copy $RX/scenes/pillow2sofa/gt_pillow2sofa $NW/gt_pillow2sofa -q --transfers 16 ) &
( [ -d $W/pgmodel/pillow2sofa_whitebg-trained ] || { rclone copyto $RX/assets/pillow2sofa_whitebg-trained.zip /root/p2s_model.zip -q && unzip -q -o /root/p2s_model.zip -d $W/pgmodel; } ) &
( [ -d $W/i-physgaussian ] || rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q ) &
( cd /dev/shm/dkta && [ -d pylib/sg ] || rclone cat $RB/code/sg_env_cf55da5.tar | tar -x ) &
if [ ! -f /root/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" h5py scikit-image lpips tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib open3d termcolor 2>&1 | tail -1 >> $L
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl >> $L 2>&1
  pip install -q pyiqa -c <(pip freeze | grep -iE "^(torch|torchvision|numpy|scipy|scikit-image|opencv-python-headless)==") >> $L 2>&1
  python -c "import torch, h5py, lpips, diff_gaussian_rasterization, simple_knn, pyiqa, open3d; print('pip ok', torch.__version__)" >> $L 2>&1 && touch /root/pip.done
fi
wait
say "환경 끝 pip $(ls /root/pip.done 2>&1) 자료 $(du -sh $NW/pillow2sofa.pt 2>&1 | cut -f1) 모델 $(ls $W/pgmodel) i-pg $(ls $W/i-physgaussian | wc -l)"
rclone copyto $RB/code/sg_train_mv.sh /root/sg_train_mv.sh -q; rclone copyto $RB/code/sg_post_p2s.sh /root/sg_post_p2s.sh -q
bash /root/sg_train_mv.sh p2s; say "학습 단계 rc=$?"
bash /root/sg_post_p2s.sh P2; say "측정 단계 rc=$?"
