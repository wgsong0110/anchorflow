#!/bin/bash
# 절 CM10 묶음 barx (vast.ai A6000 1 장 인스턴스 onstart -> tmux barx; env R2_ACCESS_KEY R2_SECRET R2_ENDPOINT AF_COMMIT, vast 가 주는 CONTAINER_ID·CONTAINER_API_KEY):
#   채택 판 (B) (장벽 + --y0eb, 절 CM8 b 판 명령: HHT ρ 0.8, K 2, 뉴턴 2, εN 1e-2 · εP 1e-1, PCG lego·낙하 4 · wolf 5, 결정적) 위에
#   d_ = + --bar_cnt (진단, 결과 불변 -- 장벽 걸린 표본점 수), x_ = + --bar_cross (관통 칸 장벽, 사용자 승인 정의 변경) + --bar_cnt, o_ = (B) 그대로 (같은 인스턴스 해시 대조).
#   씬 wolf (주 대상, --hgt 높이 곡선)·lego L000·낙하 drop1 (회귀). 그 뒤 x_wolf 덤프 -> h5 -> 3DGS 알파블렌딩 렌더 -> 정답 | 우리 합성 (exe/fem_barx_compose.py, fem_cm_wolf_cmp.mp4 형식).
# 출력 R2 lbours/fem_cm/barx/<판>_<씬>, 표 lbours/fem_cm/marks/barx_table.md, 로그 marks/barx.log, 끝 표식 marks/barx.END, 영상 result/anchorflow/v20261009/fem_barx_wolf_cmp.mp4.
# 끝나면 rclone check 차이 0 일 때 스스로 destroy (-y), 아니면 그대로 두고 marks/barx_CHECK_FAIL.
set -u
C=${AF_COMMIT:?}
R=r2:storage/result/anchorflow; MR=$R/lbours/fem_cm/marks
mkdir -p /workspace/shm /workspace/tmpd /root/bx
LOG=/root/bx/barx.log
say() { echo "[barx $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MR/barx.log -q 2>/dev/null; }
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
if [ ! -f /root/bx/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib vastai 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, h5py, matplotlib, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/bx/pip.done
fi
W=/home/dkta/work; MI=$R/_vastmirror
rclone copyto $R/vastpkg/L000.pt $W/one/L000/L000.pt -q
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q
rclone copy $MI/pgmodel/wolf_whitebg-trained $W/pgmodel/wolf_whitebg-trained --transfers 16 -q
rclone copyto $MI/PhysGaussian/config/wolf_config.json $W/PhysGaussian/config/wolf_config.json -q
rclone copy $MI/topo/pg_wolf_sand $W/topo/pg_wolf_sand --transfers 16 -q
say "[자료] $(du -sh $W/* 2>&1 | tr '\n' ' ') 정답 h5 $(ls $W/topo/pg_wolf_sand/simulation_ply | wc -l) 개"
[ -f /root/bx/pip.done ] && [ -s $W/one/L000/L000.pt ] || { say "[세팅 실패] pip $(ls /root/bx/pip.done 2>&1)"; echo "세팅 실패 $(date)" | rclone rcat $MR/barx.END; exit 1; }
# ---- 코드 (github, 인스턴스에서 편집 안 함)
[ -d /root/af/.git ] || git clone -q https://github.com/wgsong0110/anchorflow.git /root/af
git -C /root/af fetch -q origin implicit-bspline; git -C /root/af checkout -q $C || { say "[코드 실패] $C"; exit 1; }
D=/dev/shm/dkta/impwt/$C
[ -f $D/.done ] || { rm -rf $D; git clone -q /root/af $D && git -C $D checkout -q $C && echo $C > $D/.done; }
cd $D || exit 1
say "[코드] $(git log --oneline -1)"
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_barx XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
PY=/home/dkta/.conda/envs/af/bin/python
B=/dev/shm/dkta/lbours/fem_cm/barx; RR=$R/lbours/fem_cm/barx; M=/dev/shm/dkta/lbours/fem_cm/marks; J=/dev/shm/dkta/lbours/fem_cm/cm8j
mkdir -p $B $M $J $CUPY_CACHE_DIR
for j in b4_L000 b4_drop1 b5_wolf; do rclone copy $R/lbours/fem_cm/cm8j/$j $J/$j -q --exclude pred.pt; done
export CM_SCN=exe/fem_cm8_scn.sh
set +u
source exe/fem_cm8_scn.sh
for t in L000 drop1 wolf; do cm_data $t; done
table() { $PY exe/fem_barx_table.py $B $J > $M/barx_table.md 2>&1; rclone copy $M/barx_table.md $MR -q; }
run_one() { local V=$1 t=$2; shift 2
  local O=$B/${V}_$t; mkdir -p $O
  cm_scene $t "$@"
  echo "카드 0 $(cat .done) $(date '+%F %T') $SIM $DA $XA" > $O/run_info.txt
  CUDA_VISIBLE_DEVICES=0 timeout 2400 $PY -u $SIM --out $O --tag $t $DA $XA < /dev/null > $O/log.txt 2>&1
  echo "rc=$? $(date '+%F %T')" >> $O/run_info.txt
  if [ $t = drop1 ] && [ -f $O/pred.pt ]; then $PY exe/fem_ak_drop_ana.py $DD/init.pt mpm:$DD/mpm/x240.pt:0.0041666667 $V:$O/pred.pt:0.0166666667 > $O/drop_ana.txt 2>&1; fi
  say "  $V $t $(tail -1 $O/run_info.txt) $(grep -a '롤아웃\|rror\|발산' $O/log.txt | tail -2 | cut -c1-220 | tr '\n' ' ')"
  if [ $V = x ] && [ $t = wolf ]; then rclone copy $O $RR/${V}_$t -q; else rclone copy $O $RR/${V}_$t -q --exclude pred.pt; fi
  table; }
G="--ga hht --rho 0.8 --K 2 --newton 2"; T="--tolN 1e-2 --tolP 1e-1"
run_one d wolf  $G --cg 5 $T --det --hgt --bar_cnt --v=b
run_one d L000  $G --cg 4 $T --det --bar_cnt --v=b
run_one x wolf  $G --cg 5 $T --det --hgt --bar_cross --bar_cnt --dump --dump_F --v=b
run_one x L000  $G --cg 4 $T --det --bar_cross --bar_cnt --v=b
run_one x drop1 $G --cg 4 $T --det --bar_cross --bar_cnt --v=b
run_one o wolf  $G --cg 5 $T --det --hgt --v=b
run_one o L000  $G --cg 4 $T --det --v=b
run_one o drop1 $G --cg 4 $T --det --v=b
say "[판정 끝]"
# ---- wolf 영상: x_wolf 덤프 -> h5 -> 3DGS 렌더 -> 정답 | 우리 합성
O=$B/x_wolf; GT=/dev/shm/dkta/ag_data/gt_wolf; TF=/dev/shm/dkta/barxcmp; mkdir -p $TF /dev/shm/dkta/ag_data
if [ -f $O/pred.pt ]; then
  [ -f $GT/meta.json ] || CUDA_VISIBLE_DEVICES=0 $PY -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /dev/shm/dkta/ag_data/wolf_rend_tmp.pt --out_dir $GT > /dev/shm/dkta/ag_data/cv_wolf_rend.log 2>&1
  $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $GT --out $O/h5 > $O/h5.log 2>&1
  mkdir -p $O/vj
  CUDA_VISIBLE_DEVICES=0 $PY -u exe/lb_render.py --sim $O/h5 --config $GT/config.json --shape wolf --out $O/vj/fem_wolf.mp4 --frame_label 0 --caption " " < /dev/null > $O/vj/r.log 2>&1
  rclone copyto $R/lbours/fem_ag_wolf/gt/gt_wolf.mp4 $TF/gt_wolf.mp4 -q
  rclone copyto $R/lbours/fem_cm/code/NotoSansCJKkr-Bold.otf $TF/fb.otf -q; rclone copyto $R/lbours/fem_cm/code/NotoSansCJKkr-Regular.otf $TF/fr.otf -q
  if [ -f $O/vj/fem_wolf.mp4 ] && $PY exe/fem_barx_compose.py $TF/gt_wolf.mp4 $O/vj/fem_wolf.mp4 $O/summary.json $TF/fb.otf $TF/fr.otf $TF/fem_barx_wolf_cmp.mp4 $TF/txt > $TF/compose.log 2>&1; then
    rclone copyto $TF/fem_barx_wolf_cmp.mp4 $R/v20261009/fem_barx_wolf_cmp.mp4 -q && cp $TF/fem_barx_wolf_cmp.mp4 $O/vj/
    say "[영상] result/anchorflow/v20261009/fem_barx_wolf_cmp.mp4"
  else say "[영상 실패] $(tail -3 $O/vj/r.log $O/h5.log $TF/compose.log 2>/dev/null | tr '\n' ' ' | cut -c1-600)"; fi
  rclone copy $O $RR/x_wolf -q
else say "[영상 없음] x_wolf 덤프 없음"; fi
table
echo "[barx 끝] $(date '+%F %T')" > $M/barx.END; rclone copy $M/barx.END $MR -q
say "[barx 끝]"
# ---- 끝: R2 확인 차이 0 이면 destroy
rclone copy $B $RR -q --transfers 8
if rclone check $B $RR --one-way --size-only -q 2>/root/bx/check.txt; then
  say "[R2 확인 차이 0] -> destroy"
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함: $(tail -3 /root/bx/check.txt | tr '\n' ' ')"; echo "$(date) 차이 있음" | rclone rcat $MR/barx_CHECK_FAIL
fi
