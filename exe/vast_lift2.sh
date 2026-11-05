#!/bin/bash
# 절 CM11.2 묶음 lift2 (vast.ai A6000 1 장 인스턴스 onstart -> tmux lift2; env R2_ACCESS_KEY R2_SECRET R2_ENDPOINT AF_COMMIT, vast 가 주는 CONTAINER_ID·CONTAINER_API_KEY):
#   절 CM11 띄운 (B) 판 명령 그대로에서 --lift_cells 1 -> 2·3 (칸 h 의 2·3 배 = 40·60 mm) 으로 바꾸고 --bar_cross (장벽은 바닥 관통 칸 표본점에만) 를 더한 두 판 l2_wolf·l3_wolf 를 카드 0 에 동시에 (시간 측정 아님).
#   비교 = 절 CM11 l_wolf (1h (B), --bar_cross 없음; R2 lift/l_wolf 재사용) · 안 띄움 = 절 CM10 barx (x_wolf·d_wolf·o_wolf 재사용).
#   덤프 -> h5 (프레임 0 도 띄움) -> 3DGS 알파블렌딩 -> 정답 | 1h (B) | 2h bar_cross | 3h bar_cross 합성 (exe/fem_lift2_compose.py), 표 exe/fem_lift2_table.py.
# 출력 R2 lbours/fem_cm/lift2/{l2_wolf,l3_wolf}, 표 lbours/fem_cm/marks/lift2_table.md, 로그 marks/lift2.log, 끝 표식 marks/lift2.END, 영상 result/anchorflow/v20261009/fem_lift2_wolf_cmp.mp4.
# 준비 (pip·자료·코드) 를 다 끝낸 뒤 GPU 일 (판 -> 정답 변환 -> 렌더) 을 이어서. 끝나면 rclone check 차이 0 일 때 스스로 destroy (-y), 아니면 그대로 두고 marks/lift2_CHECK_FAIL.
set -u
C=${AF_COMMIT:?}
R=r2:storage/result/anchorflow; MR=$R/lbours/fem_cm/marks
mkdir -p /workspace/shm /workspace/tmpd /root/lx
LOG=/root/lx/lift2.log
say() { echo "[lift2 $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MR/lift2.log -q 2>/dev/null; }
fail() { say "$*"; echo "$* $(date)" | rclone rcat $MR/lift2.END; exit 1; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] 커밋 $C $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.draw,clocks.max.sm,memory.total --format=csv | tee -a $LOG
# ---- 클러스터 경로 그대로 (exe/vast_lift.sh 와 같은 규약)
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
PYS=$(command -v python); ln -sf $PYS /home/dkta/.conda/envs/af/bin/python
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
if [ ! -f /root/lx/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib vastai 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, h5py, matplotlib, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/lx/pip.done
fi
W=/home/dkta/work; MI=$R/_vastmirror
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q
rclone copy $MI/pgmodel/wolf_whitebg-trained $W/pgmodel/wolf_whitebg-trained --transfers 16 -q
rclone copyto $MI/PhysGaussian/config/wolf_config.json $W/PhysGaussian/config/wolf_config.json -q
rclone copy $MI/topo/pg_wolf_sand $W/topo/pg_wolf_sand --transfers 16 -q
BXL=/dev/shm/dkta/lbours/fem_cm/barx
for v in x_wolf d_wolf o_wolf; do rclone copy $R/lbours/fem_cm/barx/$v $BXL/$v -q --exclude "h5/**" --exclude h5.log; done
L1=/dev/shm/dkta/lbours/fem_cm/lift/l_wolf
rclone copy $R/lbours/fem_cm/lift/l_wolf $L1 -q --exclude "h5/**" --exclude h5.log
say "[자료] $(du -sh $W/* 2>&1 | tr '\n' ' ') 정답 h5 $(ls $W/topo/pg_wolf_sand/simulation_ply | wc -l) 개, barx $(ls $BXL/x_wolf | tr '\n' ' '), l_wolf $(ls $L1 | tr '\n' ' ')"
[ -f /root/lx/pip.done ] && [ -s $BXL/x_wolf/pred.pt ] && [ -s $L1/pred.pt ] && [ -s $L1/vj/fem_wolf_lift.mp4 ] || fail "[세팅 실패] pip $(ls /root/lx/pip.done 2>&1) barx $(ls $BXL/x_wolf 2>&1 | tr '\n' ' ') l_wolf $(ls $L1 2>&1 | tr '\n' ' ')"
# ---- 코드 (github, 인스턴스에서 편집 안 함)
[ -d /root/af/.git ] || git clone -q https://github.com/wgsong0110/anchorflow.git /root/af
git -C /root/af fetch -q origin implicit-bspline; git -C /root/af checkout -q $C || fail "[코드 실패] $C"
D=/dev/shm/dkta/impwt/$C
[ -f $D/.done ] || { rm -rf $D; git clone -q /root/af $D && git -C $D checkout -q $C && echo $C > $D/.done; }
cd $D || exit 1
say "[코드] $(git log --oneline -1)"
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_lift2 XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
PY=/home/dkta/.conda/envs/af/bin/python
B=/dev/shm/dkta/lbours/fem_cm/lift2; RR=$R/lbours/fem_cm/lift2; M=/dev/shm/dkta/lbours/fem_cm/marks
mkdir -p $B $M $CUPY_CACHE_DIR
TF=/dev/shm/dkta/lift2cmp; mkdir -p $TF
rclone copyto $R/lbours/fem_ag_wolf/gt/gt_wolf.mp4 $TF/gt_wolf.mp4 -q
rclone copyto $R/lbours/fem_cm/code/NotoSansCJKkr-Bold.otf $TF/fb.otf -q; rclone copyto $R/lbours/fem_cm/code/NotoSansCJKkr-Regular.otf $TF/fr.otf -q
export CM_SCN=exe/fem_cm8_scn.sh
set +u
source exe/fem_cm8_scn.sh
cm_data wolf
[ -s $AG/wolf.pt ] && [ -s $TF/gt_wolf.mp4 ] || fail "[세팅 실패] wolf.pt $(ls -la $AG/wolf.pt 2>&1) gt 영상 $(ls $TF 2>&1 | tr '\n' ' ')"
say "[준비 끝] GPU 일 시작"
table() { $PY exe/fem_lift2_table.py $L1 $B/l2_wolf $B/l3_wolf $BXL $AG/wolf.pt > $M/lift2_table.md 2>$M/lift2_table.err; rclone copy $M/lift2_table.md $MR -q; }
G="--ga hht --rho 0.8 --K 2 --newton 2"; T="--tolN 1e-2 --tolP 1e-1"
# 절 CM11 l_wolf 명령 그대로 + --lift_cells n (1 -> 2·3) + --bar_cross
run1() { local n=$1; local O=$B/l${n}_wolf; mkdir -p $O
  cm_scene wolf $G --cg 5 $T --det --hgt --bar_cnt --dump --dump_F --lift_cells $n --lift_auto --bar_cross --v=b
  echo "카드 0 $(cat .done) $(date '+%F %T') $SIM $DA $XA" > $O/run_info.txt
  CUDA_VISIBLE_DEVICES=0 timeout 2400 $PY -u $SIM --out $O --tag wolf $DA $XA < /dev/null > $O/log.txt 2>&1
  echo "rc=$? $(date '+%F %T')" >> $O/run_info.txt
  say "  l$n wolf $(tail -1 $O/run_info.txt) $(grep -a '절 CM11 띄움\|롤아웃\|rror\|발산' $O/log.txt | tail -3 | cut -c1-300 | tr '\n' ' ')"
  rclone copy $O $RR/l${n}_wolf -q; }
run1 2 & sleep 5; run1 3 & wait
table
# ---- 영상: 덤프 -> h5 (프레임 0 = 정답 처음 상태 + 띄운 벡터) -> 3DGS 렌더 -> 정답 | 1h (B) | 2h bar_cross | 3h bar_cross
GT=/dev/shm/dkta/ag_data/gt_wolf; mkdir -p /dev/shm/dkta/ag_data
[ -f $GT/meta.json ] || CUDA_VISIBLE_DEVICES=0 $PY -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /dev/shm/dkta/ag_data/wolf_rend_tmp.pt --out_dir $GT > /dev/shm/dkta/ag_data/cv_wolf_rend.log 2>&1
rend1() { local n=$1; local O=$B/l${n}_wolf
  [ -f $O/pred.pt ] || { say "[영상 없음] l${n}_wolf 덤프 없음"; return; }
  local OFF=$($PY -c "import json; L = json.load(open('$O/summary.json'))['lift']; print(','.join('%.9g' % (L['dist'] * q) for q in L['normal']))")
  $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $GT --out $O/h5 --x0_add "$OFF" > $O/h5.log 2>&1
  mkdir -p $O/vj
  CUDA_VISIBLE_DEVICES=0 $PY -u exe/lb_render.py --sim $O/h5 --config $GT/config.json --shape wolf --out $O/vj/fem_wolf_lift.mp4 --frame_label 0 --caption " " < /dev/null > $O/vj/r.log 2>&1
  say "  [렌더] l$n 띄움 $OFF $(ls -la $O/vj/fem_wolf_lift.mp4 2>&1 | cut -c1-200)"; }
rend1 2 & sleep 5; rend1 3 & wait
if [ -f $B/l2_wolf/vj/fem_wolf_lift.mp4 ] && [ -f $B/l3_wolf/vj/fem_wolf_lift.mp4 ] && $PY exe/fem_lift2_compose.py $TF/gt_wolf.mp4 $L1/vj/fem_wolf_lift.mp4 $L1/summary.json $B/l2_wolf/vj/fem_wolf_lift.mp4 $B/l2_wolf/summary.json $B/l3_wolf/vj/fem_wolf_lift.mp4 $B/l3_wolf/summary.json $TF/fb.otf $TF/fr.otf $TF/fem_lift2_wolf_cmp.mp4 $TF/txt > $TF/compose.log 2>&1; then
  rclone copyto $TF/fem_lift2_wolf_cmp.mp4 $R/v20261009/fem_lift2_wolf_cmp.mp4 -q && cp $TF/fem_lift2_wolf_cmp.mp4 $B/l3_wolf/vj/
  say "[영상] result/anchorflow/v20261009/fem_lift2_wolf_cmp.mp4"
else say "[영상 실패] $(for q in /dev/shm/dkta/ag_data/cv_wolf_rend.log $B/l2_wolf/h5.log $B/l2_wolf/vj/r.log $B/l3_wolf/h5.log $B/l3_wolf/vj/r.log $TF/compose.log; do [ -f $q ] && tail -n 3 $q; done | tr '\n' ' ' | cut -c1-800)"; fi
table
echo "[lift2 끝] $(date '+%F %T')" > $M/lift2.END; rclone copy $M/lift2.END $MR -q
say "[lift2 끝]"
# ---- 끝: R2 확인 차이 0 이면 destroy
rclone copy $B $RR -q --transfers 8
if rclone check $B $RR --one-way --size-only -q 2>/root/lx/check.txt; then
  say "[R2 확인 차이 0] -> destroy"
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함: $(tail -3 /root/lx/check.txt | tr '\n' ' ')"; echo "$(date) 차이 있음" | rclone rcat $MR/lift2_CHECK_FAIL
fi
