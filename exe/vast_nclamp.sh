#!/bin/bash
# 절 CM16 묶음 nclamp (vast.ai A6000 1 장 인스턴스 onstart -> tmux nclamp; env R2_ACCESS_KEY R2_SECRET R2_ENDPOINT AF_COMMIT, vast 가 주는 CONTAINER_ID·CONTAINER_API_KEY):
#   채택 판 (B) (절 CM8 b 판 명령) 에서 바닥 장벽 (--fl_bar)·띠 고착 (--fl_bst 1)·시작점 섞기 (--y0eb) 를 빼고 노드 법선 하드 클램프 --fl_nclamp (사용자 지시 하드 클램프 11-05), 결정적.
#   판 nc = 클램프, 씬 L000·0016·drop1·wolf. 대조 o_<씬> = (B) 그대로 (새 코드, 플래그 없음 + --pen 진단 -- 끄면 같은지).
#   씬 명령 = exe/fem_cm8_scn.sh cm_scene <씬> --v=a (바닥 글자 "--floor vel" = (B) 바닥 글자에서 --fl_bar --fl_bst 1 --y0eb 를 뺀 것) + --fl_nclamp (고착 KEX 0 은 플래그가 정함).
#   (A) wolf = 절 CM12 barstick a_wolf, (B) wolf 영상 = 절 CM10 barx x_wolf, 저항 강 = 절 CM15 vres vs_<씬> 재사용 (o 해시가 같으면 같은 결과).
#   영상: 덤프 -> h5 -> 3DGS 알파블렌딩 (exe/lb_render.py) 은 여기서, 합성 (exe/fem_vres_compose.py) 은 로컬 (절 CM15.3 인스턴스 합성 멈춤 때문).
# 출력 R2 lbours/fem_cm/nclamp/<판>_<씬>, 표 lbours/fem_cm/marks/nclamp_table.md, 로그 marks/nclamp.log, 끝 표식 marks/nclamp.END.
# 준비 (pip·자료·코드) 를 다 끝낸 뒤 GPU 일을 이어서. 끝나면 rclone check 차이 0 일 때 스스로 destroy (-y), 아니면 그대로 두고 marks/nclamp_CHECK_FAIL.
set -u
C=${AF_COMMIT:?}
R=r2:storage/result/anchorflow; MR=$R/lbours/fem_cm/marks
mkdir -p /workspace/shm /workspace/tmpd /root/nc
LOG=/root/nc/nclamp.log
say() { echo "[nclamp $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MR/nclamp.log -q 2>/dev/null; }
fail() { say "$*"; echo "$* $(date)" | rclone rcat $MR/nclamp.END
  [ -d /dev/shm/dkta/lbours/fem_cm/nclamp ] && rclone copy /dev/shm/dkta/lbours/fem_cm/nclamp r2:storage/result/anchorflow/lbours/fem_cm/nclamp -q --exclude pred.pt
  say "[실패 -> destroy] 자료는 R2 에"; vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG; exit 1; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git ffmpeg tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg) >/dev/null 2>&1
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] 커밋 $C $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,power.draw,clocks.max.sm,memory.total --format=csv | tee -a $LOG
say "[GPU] $(nvidia-smi --query-gpu=name,power.limit,power.default_limit --format=csv,noheader | tr '\n' ' ')"
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1); PD=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
[ -n "$PL" ] && [ -n "$PD" ] && [ "$PL" -lt "$PD" ] && { echo "전력 제한 $PL < 기본 $PD" | rclone rcat $MR/nclamp_POWERCAP; fail "[전력 제한] power.limit $PL W < 기본 $PD W -> destroy"; }
# ---- 클러스터 경로 그대로 (exe/vast_barstick.sh 와 같은 규약)
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
PYS=$(command -v python); ln -sf $PYS /home/dkta/.conda/envs/af/bin/python
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
if [ ! -f /root/nc/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib vastai 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, h5py, matplotlib, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/nc/pip.done
fi
W=/home/dkta/work; MI=$R/_vastmirror
rclone copyto $R/vastpkg/L000.pt $W/one/L000/L000.pt -q
rclone copyto $MI/lbpt/test/0016.pt $W/lbpt/test/0016.pt -q
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q
rclone copy $MI/pgmodel/wolf_whitebg-trained $W/pgmodel/wolf_whitebg-trained --transfers 16 -q
rclone copy $MI/pgmodel/lego_whitebg-trained $W/pgmodel/lego_whitebg-trained --transfers 16 -q
rclone copyto $MI/PhysGaussian/config/wolf_config.json $W/PhysGaussian/config/wolf_config.json -q
rclone copy $MI/topo/pg_wolf_sand $W/topo/pg_wolf_sand --transfers 16 -q
rclone copy $MI/lbdata $W/lbdata --transfers 16 -q
rclone copyto $MI/stab/lego/init/sim_0000000000.h5 $W/stab/lego/init/sim_0000000000.h5 -q
BXL=/dev/shm/dkta/lbours/fem_cm/barx; BSL=/dev/shm/dkta/lbours/fem_cm/barstick
rclone copy $R/lbours/fem_cm/barx/x_wolf $BXL/x_wolf -q --exclude "h5/**" --exclude h5.log
rclone copy $R/lbours/fem_cm/barstick/a_wolf $BSL/a_wolf -q --exclude "h5/**" --exclude h5.log
VRL=/dev/shm/dkta/lbours/fem_cm/vres                                               # 절 CM15 저항 강 (표·곡선; 영상은 로컬 합성)
for t in L000 0016 drop1 wolf; do rclone copy $R/lbours/fem_cm/vres/vs_$t $VRL/vs_$t -q --exclude "h5/**" --exclude h5.log --exclude "vj/**"; done
# ---- 코드 (github, 인스턴스에서 편집 안 함)
[ -d /root/af/.git ] || git clone -q https://github.com/wgsong0110/anchorflow.git /root/af
git -C /root/af fetch -q origin implicit-bspline; git -C /root/af checkout -q $C || fail "[코드 실패] $C"
D=/dev/shm/dkta/impwt/$C; [ -f $D/.done ] || { rm -rf $D; git clone -q /root/af $D && git -C $D checkout -q $C && echo $C > $D/.done; }
cd $D || exit 1
say "[코드] $(git log --oneline -1 | cut -c1-120)"
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_nc XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
PY=/home/dkta/.conda/envs/af/bin/python
B=/dev/shm/dkta/lbours/fem_cm/nclamp; RR=$R/lbours/fem_cm/nclamp; M=/dev/shm/dkta/lbours/fem_cm/marks
mkdir -p $B $M $CUPY_CACHE_DIR
export CM_SCN=exe/fem_cm8_scn.sh
set +u
source exe/fem_cm8_scn.sh
for t in L000 0016 drop1 wolf; do cm_data $t; done
say "[자료] $(du -sh $W/* 2>&1 | tr '\n' ' ') 정답 h5 $(ls $W/topo/pg_wolf_sand/simulation_ply | wc -l) 개, lego 궤적 $(ls $W/lbdata/test_low/L000 | wc -l) 개, a_wolf $(ls $BSL/a_wolf | tr '\n' ' '), x_wolf $(ls $BXL/x_wolf 2>&1 | tr '\n' ' '), vres $(ls $VRL | tr '\n' ' ')"
[ -f /root/nc/pip.done ] && [ -s $W/one/L000/L000.pt ] && [ -s $LD/0016.pt ] && [ -s $AG/wolf.pt ] && [ -s $DD/mpm/x240.pt ] && [ -s $BXL/x_wolf/summary.json ] && [ -s $BSL/a_wolf/summary.json ] && [ -s $VRL/vs_wolf/pred.pt ] && [ -s $VRL/vs_L000/summary.json ] \
  && [ -f $W/lbdata/test_low/L000/meta.json ] && [ -f $W/lbdata/config.json ] && [ -s $W/stab/lego/init/sim_0000000000.h5 ] && [ -d $W/pgmodel/lego_whitebg-trained ] \
  || fail "[세팅 실패] pip $(ls /root/nc/pip.done 2>&1) 0016 $(ls $LD 2>&1 | tr '\n' ' ') wolf $(ls $AG 2>&1 | tr '\n' ' ') drop $(ls $DD/mpm 2>&1 | tr '\n' ' ') barx $(ls $BXL/x_wolf 2>&1 | tr '\n' ' ') a $(ls $BSL/a_wolf 2>&1 | tr '\n' ' ') vres $(ls $VRL 2>&1 | tr '\n' ' ') lego $(ls $W/lbdata 2>&1 | tr '\n' ' ')"
say "[준비 끝] GPU 일 시작"
table() { $PY exe/fem_nclamp_table.py $B $VRL $BSL/a_wolf $BXL/x_wolf $AG/wolf.pt $W/PhysGaussian/config/wolf_config.json > $M/nclamp_table.md.tmp 2>$M/nclamp_table.err && mv $M/nclamp_table.md.tmp $M/nclamp_table.md; rclone copy $M/nclamp_table.md $MR -q; rclone copy $M/nclamp_table.err $MR -q; }
G="--ga hht --rho 0.8 --K 2 --newton 2"; T="--tolN 1e-2 --tolP 1e-1"
declare -A NV=([nc]="--fl_nclamp" [o]="")
declare -A FV=([nc]="--v=a" [o]="--v=b")
run_one() { local V=$1 t=$2; shift 2
  local O=$B/${V}_$t; mkdir -p $O
  ( source exe/fem_cm8_scn.sh && cm_scene $t "$@" ${NV[$V]} ${FV[$V]} && echo "카드 0 $C $(date '+%F %T') $SIM $DA $XA" > $O/run_info.txt \
    && CUDA_VISIBLE_DEVICES=0 timeout 2400 $PY -u $SIM --out $O --tag $t $DA $XA < /dev/null > $O/log.txt 2>&1; echo "rc=$? $(date '+%F %T')" >> $O/run_info.txt )
  if [ $t = drop1 ] && [ -f $O/pred.pt ]; then $PY exe/fem_ak_drop_ana.py $DD/init.pt mpm:$DD/mpm/x240.pt:0.0041666667 $V:$O/pred.pt:0.0166666667 > $O/drop_ana.txt 2>&1; fi
  say "  $V $t $(tail -1 $O/run_info.txt) $(grep -a '롤아웃\|rror\|발산' $O/log.txt | tail -2 | cut -c1-220 | tr '\n' ' ')"
  rclone copy $O $RR/${V}_$t -q
  table; }
DW="--det --hgt --pen --dump --dump_F --dump_part"
# ---- 보충 판 (env NC_ONLY="판:씬 ..."): 본 묶음 결과를 R2 에서 받고 지정 판만 카드 단독으로 다시 -> 표 -> 확인 -> destroy (연기·영상 생략)
if [ -n "${NC_ONLY:-}" ]; then
  rclone copy $RR $B -q --transfers 8 --exclude "h5/**" --exclude "vid/**" --exclude "smoke_*/**"
  say "[보충] $NC_ONLY (받은 판 $(ls $B | tr '\n' ' '))"
  for vt in $NC_ONLY; do V=${vt%%:*}; t=${vt#*:}; rm -rf $B/${V}_$t
    case $t in wolf|L000) run_one $V $t $G --cg $([ $t = wolf ] && echo 5 || echo 4) $T $DW;; *) run_one $V $t $G --cg 4 $T --det --pen;; esac; done
  table; echo "[nclamp 끝 (보충 $NC_ONLY)] $(date '+%F %T')" > $M/nclamp.END; rclone copy $M/nclamp.END $MR -q
  if rclone check $B $RR --one-way --size-only -q 2>/root/nc/check.txt; then say "[R2 확인 차이 0] -> destroy"; vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
  else say "[R2 확인 차이 있음] destroy 안 함: $(tail -3 /root/nc/check.txt | tr '\n' ' ')"; echo "$(date) 차이 있음" | rclone rcat $MR/nclamp_CHECK_FAIL; fi
  exit 0
fi
# ---- 연기 (3 프레임): 새 커널 컴파일·실행 확인 (결정적 클램프 L000·wolf, 비결정 클램프 L000). 실패하면 기록 올리고 destroy
smk() { local V=$1 t=$2; shift 2; local O=$B/smoke_${V}_$t$2; mkdir -p $O
  ( source exe/fem_cm8_scn.sh && cm_scene $t $1 ${NV[$V]} ${FV[$V]} && echo "$SIM $DA $XA" > $O/cmd.txt && CUDA_VISIBLE_DEVICES=0 timeout 900 $PY -u $SIM --out $O --tag $t $DA $XA --frames 3 < /dev/null > $O/log.txt 2>&1; echo "rc=$?" > $O/rc.txt )
  grep -q '"ncl"' $O/fem.jsonl 2>/dev/null && [ "$(cat $O/rc.txt)" = rc=0 ] && [ -f $O/summary.json ] && grep -q -- '--fl_nclamp' $O/cmd.txt && ! grep -q -- '--fl_bar\|--y0eb\|--fl_bst\|--fl_vres' $O/cmd.txt; }
smk nc L000 "$G --cg 4 $T $DW" "" && smk nc wolf "$G --cg 5 $T $DW" "" && smk nc L000 "$G --cg 4 $T --pen" _nd \
  && say "[연기 통과] $(for q in $B/smoke_*; do echo "$(basename $q) $(tail -1 $q/fem.jsonl | grep -o '"pen": {[^}]*}\|"ncl": {[^}]*}' | tr '\n' ' ')"; done | tr '\n' ' ')" \
  || fail "[연기 실패] $(for q in $B/smoke_*; do echo "$(basename $q) $(cat $q/rc.txt 2>/dev/null) $(grep -a 'rror\|Trace' $q/log.txt | tail -3 | cut -c1-300)"; done | tr '\n' ' ')"
say "[명령 확인] 클램프 L000: $(cat $B/smoke_nc_L000/cmd.txt)"
lane1() { run_one nc wolf $G --cg 5 $T $DW; run_one o wolf $G --cg 5 $T --det --hgt --pen; }
lane2() { run_one nc L000 $G --cg 4 $T $DW; run_one o L000 $G --cg 4 $T --det --pen; }
lane3() { for V in nc o; do run_one $V drop1 $G --cg 4 $T --det --pen; run_one $V 0016 $G --cg 4 $T --det --pen; done; }
lane1 & sleep 5; lane2 & sleep 5; lane3 & wait
# GPU 메모리 부족 (세 줄 동시, 절 CM15.3 vw_0016) 으로 실패한 판은 단독으로 다시 (--dump_part 판은 summary.json 의 oom, 아니면 log 의 메모리 부족)
args_of() { case $1_$2 in nc_wolf) echo "$G --cg 5 $T $DW";; nc_L000) echo "$G --cg 4 $T $DW";; o_wolf) echo "$G --cg 5 $T --det --hgt --pen";; *) echo "$G --cg 4 $T --det --pen";; esac; }
for O in $B/nc_* $B/o_*; do
  if [ -f $O/summary.json ]; then grep -q '"oom"' $O/summary.json || continue; else grep -aq "OutOfMemory\|out of memory" $O/log.txt 2>/dev/null || continue; fi
  V=$(basename $O); t=${V#*_}; V=${V%%_*}; say "[메모리 부족 -> 단독 다시] ${V}_$t"; rm -rf $O
  run_one $V $t $(args_of $V $t); done
say "[판정 끝] 해시 o_wolf $(grep -o '"x_sha1": "[0-9a-f]*"' $B/o_wolf/summary.json) (9b119d8fd3) o_L000 $(grep -o '"x_sha1": "[0-9a-f]*"' $B/o_L000/summary.json) (c43a42c7d0) o_drop1 $(grep -o '"x_sha1": "[0-9a-f]*"' $B/o_drop1/summary.json) (de2e854bf7) o_0016 $(grep -o '"x_sha1": "[0-9a-f]*"' $B/o_0016/summary.json) (f89f6fb659)"
table
# ---- 영상: 덤프 -> h5 -> 3DGS 렌더
GT=/dev/shm/dkta/ag_data/gt_wolf; mkdir -p /dev/shm/dkta/ag_data
[ -f $GT/meta.json ] || CUDA_VISIBLE_DEVICES=0 $PY -u exe/pg_h5_to_pt.py --h5_dir $W/topo/pg_wolf_sand --config $W/PhysGaussian/config/wolf_config.json --out_pt /dev/shm/dkta/ag_data/wolf_rend_tmp.pt --out_dir $GT > /dev/shm/dkta/ag_data/cv_wolf_rend.log 2>&1
rend1() { local V=$1 t=$2 O=$B/$1_$2 sh
  [ -f $O/pred.pt ] || { say "[영상 없음] ${V}_$t 덤프 없음"; return; }
  if [ $t = wolf ]; then sh=wolf; $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $GT --out $O/h5 > $O/h5.log 2>&1; CF=$GT/config.json
  else sh=lego; $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $W/lbdata/test_low/L000 --out $O/h5 > $O/h5.log 2>&1; CF=$W/lbdata/config.json; fi
  mkdir -p $O/vj
  CUDA_VISIBLE_DEVICES=0 $PY -u exe/lb_render.py --sim $O/h5 --config $CF --shape $sh --out $O/vj/fem_$sh.mp4 --frame_label 0 --caption " " < /dev/null > $O/vj/r.log 2>&1
  say "  [렌더] ${V}_$t $(ls -la $O/vj/fem_$sh.mp4 2>&1 | cut -c1-200) $(tail -1 $O/vj/r.log | cut -c1-200)"
  rclone copy $O $RR/${V}_$t -q --exclude pred.pt; }
rend1 nc wolf & sleep 3; rend1 nc L000 & wait
table
echo "[nclamp 끝] $(date '+%F %T')" > $M/nclamp.END; rclone copy $M/nclamp.END $MR -q
say "[nclamp 끝]"
# ---- 끝: R2 확인 차이 0 이면 destroy
rclone copy $B $RR -q --transfers 8
if rclone check $B $RR --one-way --size-only -q 2>/root/nc/check.txt; then
  say "[R2 확인 차이 0] -> destroy"
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함: $(tail -3 /root/nc/check.txt | tr '\n' ' ')"; echo "$(date) 차이 있음" | rclone rcat $MR/nclamp_CHECK_FAIL
fi
