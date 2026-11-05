#!/bin/bash
# 절 CM12 묶음 barstick (vast.ai A6000 1 장 인스턴스 onstart -> tmux barstick; env R2_ACCESS_KEY R2_SECRET R2_ENDPOINT AF_COMMIT AF_OLD, vast 가 주는 CONTAINER_ID·CONTAINER_API_KEY):
#   장벽 바닥 (B) (절 CM8 b 판 명령: 장벽 + --y0eb) 에 (A) 가우스 점 고착 (절 CM8 a 판 정의) 을 함께 켠 판 --fl_stk (사용자 승인 11-05 정의 변경), 결정적.
#   판: s_wolf (안 띄운 원래 초기 상태, --hgt --bar_cnt --dump --dump_F)·s_L000·s_0016·s_drop1 (회귀), 대조 o_* = (B) 그대로 (새 코드, 플래그 없음),
#       o0_L000 = 옛 코드 (AF_OLD 커밋) 로 같은 (B) 명령 -- 끄면 비트 같은지, a_wolf = (A) 절 CM8 a3 명령 (--v=a --cg 3) 다시 (+ --hgt --dump, 높이 곡선·영상용).
#   (B) wolf 높이 곡선·영상 = 절 CM10 barx x_wolf (= barx o_wolf 비트 같음) 재사용.
#   영상: s_wolf·a_wolf 덤프 -> h5 -> 3DGS 알파블렌딩 -> 정답 | (A) | (B) | (B)+고착 (exe/fem_barstick_compose.py) -> result/anchorflow/v20261009/fem_barstick_wolf_cmp.mp4.
# 출력 R2 lbours/fem_cm/barstick/<판>_<씬>, 표 lbours/fem_cm/marks/barstick_table.md, 로그 marks/barstick.log, 끝 표식 marks/barstick.END.
# 준비 (pip·자료·코드) 를 다 끝낸 뒤 GPU 일을 이어서. 끝나면 rclone check 차이 0 일 때 스스로 destroy (-y), 아니면 그대로 두고 marks/barstick_CHECK_FAIL.
set -u
C=${AF_COMMIT:?}; C0=${AF_OLD:?}
R=r2:storage/result/anchorflow; MR=$R/lbours/fem_cm/marks
mkdir -p /workspace/shm /workspace/tmpd /root/bs
LOG=/root/bs/barstick.log
say() { echo "[barstick $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MR/barstick.log -q 2>/dev/null; }
fail() { say "$*"; echo "$* $(date)" | rclone rcat $MR/barstick.END; exit 1; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] 커밋 $C (옛 $C0) $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,power.draw,clocks.max.sm,memory.total --format=csv | tee -a $LOG
# ---- 클러스터 경로 그대로 (exe/vast_lift2.sh 와 같은 규약)
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
PYS=$(command -v python); ln -sf $PYS /home/dkta/.conda/envs/af/bin/python
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
if [ ! -f /root/bs/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib vastai 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, h5py, matplotlib, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/bs/pip.done
fi
W=/home/dkta/work; MI=$R/_vastmirror
rclone copyto $R/vastpkg/L000.pt $W/one/L000/L000.pt -q
rclone copyto $MI/lbpt/test/0016.pt $W/lbpt/test/0016.pt -q
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q
rclone copy $MI/pgmodel/wolf_whitebg-trained $W/pgmodel/wolf_whitebg-trained --transfers 16 -q
rclone copyto $MI/PhysGaussian/config/wolf_config.json $W/PhysGaussian/config/wolf_config.json -q
rclone copy $MI/topo/pg_wolf_sand $W/topo/pg_wolf_sand --transfers 16 -q
BXL=/dev/shm/dkta/lbours/fem_cm/barx
rclone copy $R/lbours/fem_cm/barx/x_wolf $BXL/x_wolf -q --exclude "h5/**" --exclude h5.log
for v in o_wolf o_L000 o_drop1; do rclone copy $R/lbours/fem_cm/barx/$v $BXL/$v -q --exclude pred.pt; done
# ---- 코드 (github, 인스턴스에서 편집 안 함): 새 커밋 $C, 끄면 같은지 대조용 옛 커밋 $C0
[ -d /root/af/.git ] || git clone -q https://github.com/wgsong0110/anchorflow.git /root/af
git -C /root/af fetch -q origin implicit-bspline; git -C /root/af checkout -q $C || fail "[코드 실패] $C"
for c in $C $C0; do D_=/dev/shm/dkta/impwt/$c; [ -f $D_/.done ] || { rm -rf $D_; git clone -q /root/af $D_ && git -C $D_ checkout -q $c && echo $c > $D_/.done; }; done
D=/dev/shm/dkta/impwt/$C; D0=/dev/shm/dkta/impwt/$C0
cd $D || exit 1
say "[코드] $(git log --oneline -1) / 옛 $(git -C $D0 log --oneline -1)"
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_bs XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
PY=/home/dkta/.conda/envs/af/bin/python
B=/dev/shm/dkta/lbours/fem_cm/barstick; RR=$R/lbours/fem_cm/barstick; M=/dev/shm/dkta/lbours/fem_cm/marks
mkdir -p $B $M $CUPY_CACHE_DIR
TF=/dev/shm/dkta/bscmp; mkdir -p $TF
rclone copyto $R/lbours/fem_ag_wolf/gt/gt_wolf.mp4 $TF/gt_wolf.mp4 -q
rclone copyto $R/lbours/fem_cm/code/NotoSansCJKkr-Bold.otf $TF/fb.otf -q; rclone copyto $R/lbours/fem_cm/code/NotoSansCJKkr-Regular.otf $TF/fr.otf -q
export CM_SCN=exe/fem_cm8_scn.sh
set +u
source exe/fem_cm8_scn.sh
for t in L000 0016 drop1 wolf; do cm_data $t; done
say "[자료] $(du -sh $W/* 2>&1 | tr '\n' ' ') 정답 h5 $(ls $W/topo/pg_wolf_sand/simulation_ply | wc -l) 개, 0016 $(ls -la $LD/0016.pt 2>&1 | cut -c1-120), barx x_wolf $(ls $BXL/x_wolf | tr '\n' ' ') vj $(ls $BXL/x_wolf/vj 2>&1 | tr '\n' ' ')"
[ -f /root/bs/pip.done ] && [ -s $W/one/L000/L000.pt ] && [ -s $LD/0016.pt ] && [ -s $AG/wolf.pt ] && [ -s $DD/mpm/x240.pt ] && [ -s $BXL/x_wolf/vj/fem_wolf.mp4 ] && [ -s $TF/gt_wolf.mp4 ] \
  || fail "[세팅 실패] pip $(ls /root/bs/pip.done 2>&1) 0016 $(ls $LD 2>&1 | tr '\n' ' ') wolf $(ls $AG 2>&1 | tr '\n' ' ') drop $(ls $DD/mpm 2>&1 | tr '\n' ' ') barx $(ls $BXL/x_wolf/vj 2>&1 | tr '\n' ' ')"
say "[준비 끝] GPU 일 시작"
table() { $PY exe/fem_barstick_table.py $B $BXL $AG/wolf.pt > $M/barstick_table.md 2>$M/barstick_table.err; rclone copy $M/barstick_table.md $MR -q; }
G="--ga hht --rho 0.8 --K 2 --newton 2"; T="--tolN 1e-2 --tolP 1e-1"
run_one() { local V=$1 t=$2 DIR=$3; shift 3
  local O=$B/${V}_$t; mkdir -p $O
  ( cd $DIR && source exe/fem_cm8_scn.sh && cm_scene $t "$@" && echo "카드 0 $(cat .done) $(date '+%F %T') $SIM $DA $XA" > $O/run_info.txt \
    && CUDA_VISIBLE_DEVICES=0 timeout 2400 $PY -u $SIM --out $O --tag $t $DA $XA < /dev/null > $O/log.txt 2>&1; echo "rc=$? $(date '+%F %T')" >> $O/run_info.txt )
  if [ $t = drop1 ] && [ -f $O/pred.pt ]; then $PY exe/fem_ak_drop_ana.py $DD/init.pt mpm:$DD/mpm/x240.pt:0.0041666667 $V:$O/pred.pt:0.0166666667 > $O/drop_ana.txt 2>&1; fi
  say "  $V $t $(tail -1 $O/run_info.txt) $(grep -a '롤아웃\|rror\|발산' $O/log.txt | tail -2 | cut -c1-220 | tr '\n' ' ')"
  if [ $t = wolf ] && [ $V != o ]; then rclone copy $O $RR/${V}_$t -q; else rclone copy $O $RR/${V}_$t -q --exclude pred.pt; fi
  table; }
lane1() {
  run_one s wolf $D $G --cg 5 $T --det --hgt --bar_cnt --dump --dump_F --fl_stk --v=b
  run_one a wolf $D $G --cg 3 $T --det --hgt --dump --dump_F --v=a
  run_one o wolf $D $G --cg 5 $T --det --hgt --v=b; }
lane2() {
  run_one s L000  $D $G --cg 4 $T --det --fl_stk --v=b
  run_one s 0016  $D $G --cg 4 $T --det --fl_stk --v=b
  run_one s drop1 $D $G --cg 4 $T --det --fl_stk --v=b
  run_one o L000  $D $G --cg 4 $T --det --v=b
  run_one o0 L000 $D0 $G --cg 4 $T --det --v=b
  run_one o 0016  $D $G --cg 4 $T --det --v=b
  run_one o drop1 $D $G --cg 4 $T --det --v=b; }
lane1 & sleep 5; lane2 & wait
say "[판정 끝] 해시 o_L000 $(grep -o '"x_sha1": "[0-9a-f]*"' $B/o_L000/summary.json) o0_L000 $(grep -o '"x_sha1": "[0-9a-f]*"' $B/o0_L000/summary.json)"
table
# ---- 영상: 덤프 -> h5 -> 3DGS 렌더 -> 정답 | (A) | (B) (barx x_wolf) | (B)+고착
GT=/dev/shm/dkta/ag_data/gt_wolf; mkdir -p /dev/shm/dkta/ag_data
[ -f $GT/meta.json ] || CUDA_VISIBLE_DEVICES=0 $PY -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /dev/shm/dkta/ag_data/wolf_rend_tmp.pt --out_dir $GT > /dev/shm/dkta/ag_data/cv_wolf_rend.log 2>&1
rend1() { local O=$B/$1_wolf
  [ -f $O/pred.pt ] || { say "[영상 없음] $1_wolf 덤프 없음"; return; }
  $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $GT --out $O/h5 > $O/h5.log 2>&1
  mkdir -p $O/vj
  CUDA_VISIBLE_DEVICES=0 $PY -u exe/lb_render.py --sim $O/h5 --config $GT/config.json --shape wolf --out $O/vj/fem_wolf.mp4 --frame_label 0 --caption " " < /dev/null > $O/vj/r.log 2>&1
  say "  [렌더] $1 $(ls -la $O/vj/fem_wolf.mp4 2>&1 | cut -c1-200)"; }
rend1 s & sleep 5; rend1 a & wait
OUTV=$R/v20261009/fem_barstick_wolf_cmp.mp4
if [ -f $B/s_wolf/vj/fem_wolf.mp4 ] && [ -f $B/a_wolf/vj/fem_wolf.mp4 ] && $PY exe/fem_barstick_compose.py $TF/gt_wolf.mp4 $B/a_wolf/vj/fem_wolf.mp4 $B/a_wolf/summary.json $BXL/x_wolf/vj/fem_wolf.mp4 $BXL/x_wolf/summary.json $B/s_wolf/vj/fem_wolf.mp4 $B/s_wolf/summary.json $TF/fb.otf $TF/fr.otf $TF/fem_barstick_wolf_cmp.mp4 $TF/txt > $TF/compose.log 2>&1; then
  cp $TF/fem_barstick_wolf_cmp.mp4 $B/s_wolf/vj/
  rclone copyto $TF/fem_barstick_wolf_cmp.mp4 $OUTV -q && rclone check $TF $R/v20261009 --include fem_barstick_wolf_cmp.mp4 --one-way --size-only -q \
    && say "[영상] result/anchorflow/v20261009/fem_barstick_wolf_cmp.mp4 (R2 확인)" || say "[영상 올리기 확인 실패]"
else say "[영상 실패] $(for q in /dev/shm/dkta/ag_data/cv_wolf_rend.log $B/s_wolf/h5.log $B/s_wolf/vj/r.log $B/a_wolf/h5.log $B/a_wolf/vj/r.log $TF/compose.log; do [ -f $q ] && tail -n 3 $q; done | tr '\n' ' ' | cut -c1-800)"; fi
table
echo "[barstick 끝] $(date '+%F %T')" > $M/barstick.END; rclone copy $M/barstick.END $MR -q
say "[barstick 끝]"
# ---- 끝: R2 확인 차이 0 이면 destroy
rclone copy $B $RR -q --transfers 8
if rclone check $B $RR --one-way --size-only -q 2>/root/bs/check.txt; then
  say "[R2 확인 차이 0] -> destroy"
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함: $(tail -3 /root/bs/check.txt | tr '\n' ' ')"; echo "$(date) 차이 있음" | rclone rcat $MR/barstick_CHECK_FAIL
fi
