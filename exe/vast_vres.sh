#!/bin/bash
# 절 CM15 묶음 vres (vast.ai A6000 1 장 인스턴스 onstart -> tmux vres; env R2_ACCESS_KEY R2_SECRET R2_ENDPOINT AF_COMMIT, vast 가 주는 CONTAINER_ID·CONTAINER_API_KEY):
#   채택 판 (B) (절 CM8 b 판 명령) 에서 바닥 장벽 (--fl_bar)·띠 고착 (--fl_bst 1)·시작점 섞기 (--y0eb) 를 빼고 −z 저항 바닥 --fl_vres k (사용자 승인 11-05 정의 변경), 결정적.
#   판 = vs (강, k 9: 관성만이면 한 걸음 법선 이동의 1/10 만 남김) / vw (약, k 0.9 = 강의 1/10), 씬 L000·0016·drop1·wolf. 대조 o_<씬> = (B) 그대로 (새 코드, 플래그 없음 + --pen 진단 -- 끄면 같은지).
#   씬 명령 = exe/fem_cm8_scn.sh cm_scene <씬> --v=a (바닥 글자 "--floor vel" = (B) 바닥 글자에서 --fl_bar --fl_bst 1 --y0eb 를 뺀 것) + --fl_vres k.
#   (A) wolf 곡선·영상 = 절 CM12 barstick a_wolf, (B) wolf 영상 = 절 CM10 barx x_wolf 재사용 (o_wolf 해시가 같으면 같은 결과).
#   영상: 덤프 -> h5 -> 3DGS 알파블렌딩 -> lego L000 (정답 | (B) | 강 | 약), wolf (정답 | (A) | (B) | 강 | 약) (exe/fem_vres_compose.py, 가로 4096 이하·yuv420p·faststart)
#     -> result/anchorflow/v20261009/fem_vres_{lego_L000,wolf}_cmp.mp4.
# 출력 R2 lbours/fem_cm/vres/<판>_<씬>, 표 lbours/fem_cm/marks/vres_table.md, 로그 marks/vres.log, 끝 표식 marks/vres.END.
# 준비 (pip·자료·코드) 를 다 끝낸 뒤 GPU 일을 이어서. 끝나면 rclone check 차이 0 일 때 스스로 destroy (-y), 아니면 그대로 두고 marks/vres_CHECK_FAIL.
set -u
C=${AF_COMMIT:?}
R=r2:storage/result/anchorflow; MR=$R/lbours/fem_cm/marks
mkdir -p /workspace/shm /workspace/tmpd /root/vr
LOG=/root/vr/vres.log
say() { echo "[vres $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MR/vres.log -q 2>/dev/null; }
fail() { say "$*"; echo "$* $(date)" | rclone rcat $MR/vres.END
  [ -d /dev/shm/dkta/lbours/fem_cm/vres ] && rclone copy /dev/shm/dkta/lbours/fem_cm/vres r2:storage/result/anchorflow/lbours/fem_cm/vres -q --exclude pred.pt
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
[ -n "$PL" ] && [ -n "$PD" ] && [ "$PL" -lt "$PD" ] && { echo "전력 제한 $PL < 기본 $PD" | rclone rcat $MR/vres_POWERCAP; fail "[전력 제한] power.limit $PL W < 기본 $PD W -> destroy"; }
# ---- 클러스터 경로 그대로 (exe/vast_barstick.sh 와 같은 규약)
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
PYS=$(command -v python); ln -sf $PYS /home/dkta/.conda/envs/af/bin/python
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
if [ ! -f /root/vr/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib vastai 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, h5py, matplotlib, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/vr/pip.done
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
# ---- 코드 (github, 인스턴스에서 편집 안 함)
[ -d /root/af/.git ] || git clone -q https://github.com/wgsong0110/anchorflow.git /root/af
git -C /root/af fetch -q origin implicit-bspline; git -C /root/af checkout -q $C || fail "[코드 실패] $C"
D=/dev/shm/dkta/impwt/$C; [ -f $D/.done ] || { rm -rf $D; git clone -q /root/af $D && git -C $D checkout -q $C && echo $C > $D/.done; }
cd $D || exit 1
say "[코드] $(git log --oneline -1 | cut -c1-120)"
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_vr XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
PY=/home/dkta/.conda/envs/af/bin/python
B=/dev/shm/dkta/lbours/fem_cm/vres; RR=$R/lbours/fem_cm/vres; M=/dev/shm/dkta/lbours/fem_cm/marks
mkdir -p $B $M $CUPY_CACHE_DIR
TF=/dev/shm/dkta/vrcmp; mkdir -p $TF
rclone copyto $R/lbours/fem_ag_wolf/gt/gt_wolf.mp4 $TF/gt_wolf.mp4 -q
rclone copyto $R/lbours/fem2_lego/gt_lego.mp4 $TF/gt_L000.mp4 -q
rclone copyto $R/lbours/fem_cm/code/NotoSansCJKkr-Bold.otf $TF/fb.otf -q; rclone copyto $R/lbours/fem_cm/code/NotoSansCJKkr-Regular.otf $TF/fr.otf -q
export CM_SCN=exe/fem_cm8_scn.sh
set +u
source exe/fem_cm8_scn.sh
for t in L000 0016 drop1 wolf; do cm_data $t; done
say "[자료] $(du -sh $W/* 2>&1 | tr '\n' ' ') 정답 h5 $(ls $W/topo/pg_wolf_sand/simulation_ply | wc -l) 개, lego 궤적 $(ls $W/lbdata/test_low/L000 | wc -l) 개, a_wolf $(ls $BSL/a_wolf | tr '\n' ' '), x_wolf vj $(ls $BXL/x_wolf/vj 2>&1 | tr '\n' ' ')"
[ -f /root/vr/pip.done ] && [ -s $W/one/L000/L000.pt ] && [ -s $LD/0016.pt ] && [ -s $AG/wolf.pt ] && [ -s $DD/mpm/x240.pt ] && [ -s $BXL/x_wolf/vj/fem_wolf.mp4 ] && [ -s $BSL/a_wolf/vj/fem_wolf.mp4 ] \
  && [ -s $TF/gt_wolf.mp4 ] && [ -s $TF/gt_L000.mp4 ] && [ -f $W/lbdata/test_low/L000/meta.json ] && [ -f $W/lbdata/config.json ] && [ -s $W/stab/lego/init/sim_0000000000.h5 ] && [ -d $W/pgmodel/lego_whitebg-trained ] \
  || fail "[세팅 실패] pip $(ls /root/vr/pip.done 2>&1) 0016 $(ls $LD 2>&1 | tr '\n' ' ') wolf $(ls $AG 2>&1 | tr '\n' ' ') drop $(ls $DD/mpm 2>&1 | tr '\n' ' ') barx $(ls $BXL/x_wolf/vj 2>&1 | tr '\n' ' ') a $(ls $BSL/a_wolf/vj 2>&1 | tr '\n' ' ') lego $(ls $W/lbdata 2>&1 | tr '\n' ' ')"
say "[준비 끝] GPU 일 시작"
table() { $PY exe/fem_vres_table.py $B $BSL/a_wolf $BXL/x_wolf $AG/wolf.pt $W/PhysGaussian/config/wolf_config.json > $M/vres_table.md.tmp 2>$M/vres_table.err && mv $M/vres_table.md.tmp $M/vres_table.md; rclone copy $M/vres_table.md $MR -q; rclone copy $M/vres_table.err $MR -q; }
G="--ga hht --rho 0.8 --K 2 --newton 2"; T="--tolN 1e-2 --tolP 1e-1"
declare -A NV=([vs]="--fl_vres 9" [vw]="--fl_vres 0.9" [o]="")
declare -A FV=([vs]="--v=a" [vw]="--v=a" [o]="--v=b")
run_one() { local V=$1 t=$2; shift 2
  local O=$B/${V}_$t; mkdir -p $O
  ( source exe/fem_cm8_scn.sh && cm_scene $t "$@" ${NV[$V]} ${FV[$V]} && echo "카드 0 $C $(date '+%F %T') $SIM $DA $XA" > $O/run_info.txt \
    && CUDA_VISIBLE_DEVICES=0 timeout 2400 $PY -u $SIM --out $O --tag $t $DA $XA < /dev/null > $O/log.txt 2>&1; echo "rc=$? $(date '+%F %T')" >> $O/run_info.txt )
  if [ $t = drop1 ] && [ -f $O/pred.pt ]; then $PY exe/fem_ak_drop_ana.py $DD/init.pt mpm:$DD/mpm/x240.pt:0.0041666667 $V:$O/pred.pt:0.0166666667 > $O/drop_ana.txt 2>&1; fi
  say "  $V $t $(tail -1 $O/run_info.txt) $(grep -a '롤아웃\|rror\|발산' $O/log.txt | tail -2 | cut -c1-220 | tr '\n' ' ')"
  rclone copy $O $RR/${V}_$t -q
  table; }
DW="--det --hgt --pen --dump --dump_F --dump_part"
# ---- 보충 판 (env VR_ONLY="판:씬 ...", 절 CM15.3): 본 묶음 결과를 R2 에서 받고 지정 판만 카드 단독으로 다시 -> 표 -> 확인 -> destroy (연기·영상 생략)
if [ -n "${VR_ONLY:-}" ]; then
  rclone copy $RR $B -q --transfers 8 --exclude "h5/**" --exclude "vid/**" --exclude "smoke_*/**"
  say "[보충] $VR_ONLY (받은 판 $(ls $B | tr '\n' ' '))"
  for vt in $VR_ONLY; do V=${vt%%:*}; t=${vt#*:}; rm -rf $B/${V}_$t
    case $t in wolf|L000) run_one $V $t $G --cg $([ $t = wolf ] && echo 5 || echo 4) $T $DW;; *) run_one $V $t $G --cg 4 $T --det --pen;; esac; done
  table; echo "[vres 끝 (보충 $VR_ONLY)] $(date '+%F %T')" > $M/vres.END; rclone copy $M/vres.END $MR -q
  if rclone check $B $RR --one-way --size-only -q 2>/root/vr/check.txt; then say "[R2 확인 차이 0] -> destroy"; vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
  else say "[R2 확인 차이 있음] destroy 안 함: $(tail -3 /root/vr/check.txt | tr '\n' ' ')"; echo "$(date) 차이 있음" | rclone rcat $MR/vres_CHECK_FAIL; fi
  exit 0
fi
# ---- 연기 (3 프레임): 새 커널 컴파일·실행 확인 (결정적 강 L000·약 wolf, 비결정 강 L000). 실패하면 기록 올리고 destroy
smk() { local V=$1 t=$2; shift 2; local O=$B/smoke_${V}_$t$2; mkdir -p $O
  ( source exe/fem_cm8_scn.sh && cm_scene $t $1 ${NV[$V]} ${FV[$V]} && echo "$SIM $DA $XA" > $O/cmd.txt && CUDA_VISIBLE_DEVICES=0 timeout 900 $PY -u $SIM --out $O --tag $t $DA $XA --frames 3 < /dev/null > $O/log.txt 2>&1; echo "rc=$?" > $O/rc.txt )
  grep -q '"nq"' $O/fem.jsonl 2>/dev/null && [ "$(cat $O/rc.txt)" = rc=0 ] && [ -f $O/summary.json ] && ! grep -q -- '--fl_bar\|--y0eb\|--fl_bst' $O/cmd.txt; }
smk vs L000 "$G --cg 4 $T $DW" "" && smk vw wolf "$G --cg 5 $T $DW" "" && smk vs L000 "$G --cg 4 $T --pen" _nd \
  && say "[연기 통과] $(for q in $B/smoke_*; do echo "$(basename $q) $(cut -c1-60 $q/cmd.txt | head -1) $(tail -1 $q/fem.jsonl | grep -o '"pen": {[^}]*}')"; done | tr '\n' ' ')" \
  || fail "[연기 실패] $(for q in $B/smoke_*; do echo "$(basename $q) $(cat $q/rc.txt 2>/dev/null) $(grep -a 'rror\|Trace' $q/log.txt | tail -3 | cut -c1-300)"; done | tr '\n' ' ')"
say "[명령 확인] 강 L000: $(cat $B/smoke_vs_L000/cmd.txt)"
lane1() { for V in vs vw; do run_one $V wolf $G --cg 5 $T $DW; done; run_one o wolf $G --cg 5 $T --det --hgt --pen; }
lane2() { for V in o vs vw; do run_one $V L000 $G --cg 4 $T $DW; done; }
lane3() { for V in vs vw o; do run_one $V drop1 $G --cg 4 $T --det --pen; run_one $V 0016 $G --cg 4 $T --det --pen; done; }
lane1 & sleep 5; lane2 & sleep 5; lane3 & wait
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
for V in vs vw; do rend1 $V wolf & sleep 3; rend1 $V L000 & wait; done
rend1 o L000
CAP="HHT ρ 0.8, K 2, 뉴턴 2, εN 1e-2 · εP 1e-1, 결정적 · (B) = 장벽 + 띠 고착 + --y0eb · −z 저항 바닥 = 장벽·고착 없이 바닥 밑 표본점의 법선 반대 방향 이동에 ½ c m s², c = k/(τ² thinv²), 접선 자유"
NM() { case $1 in vs) echo "−z 저항 강 (k 9)";; vw) echo "−z 저항 약 (k 0.9)";; esac; }
ARGL=("정답 (MPM)" $TF/gt_L000.mp4 - "(B) 장벽 (표본점)" $B/o_L000/vj/fem_lego.mp4 $B/o_L000/summary.json)
ARGW=("정답 (PG 모래)" $TF/gt_wolf.mp4 - "(A) 가우스 점 고착" $BSL/a_wolf/vj/fem_wolf.mp4 $BSL/a_wolf/summary.json "(B) 장벽 (표본점)" $BXL/x_wolf/vj/fem_wolf.mp4 $BXL/x_wolf/summary.json)
for V in vs vw; do ARGL+=("$(NM $V)" $B/${V}_L000/vj/fem_lego.mp4 $B/${V}_L000/summary.json); ARGW+=("$(NM $V)" $B/${V}_wolf/vj/fem_wolf.mp4 $B/${V}_wolf/summary.json); done
for t in L000 wolf; do
  if [ $t = L000 ]; then NAME=fem_vres_lego_L000_cmp.mp4; TT="lego L000 80 프레임 · PCG 상한 4 · $CAP"; A=("${ARGL[@]}"); else NAME=fem_vres_wolf_cmp.mp4; TT="wolf 50 프레임 · PCG 상한 5 ((A) 3) · $CAP"; A=("${ARGW[@]}"); fi
  if timeout 1200 $PY exe/fem_vres_compose.py $TF/$NAME "$TT" $TF/fb.otf $TF/fr.otf $TF/tmp_$t "${A[@]}" > $TF/compose_$t.log 2>&1; then
    rclone copyto $TF/$NAME $R/v20261009/$NAME -q && rclone check $TF $R/v20261009 --include $NAME --one-way --size-only -q \
      && say "[영상] result/anchorflow/v20261009/$NAME (R2 확인) $(ffprobe -v error -show_entries stream=width,height,pix_fmt,nb_frames -of csv $TF/$NAME | tr '\n' ' ')" || say "[영상 올리기 확인 실패] $NAME"
    rclone copyto $TF/$NAME $RR/vid/$NAME -q
  else say "[영상 실패] $t $(tail -5 $TF/compose_$t.log | tr '\n' ' ' | cut -c1-600)"; fi
done
table
echo "[vres 끝] $(date '+%F %T')" > $M/vres.END; rclone copy $M/vres.END $MR -q
say "[vres 끝]"
# ---- 끝: R2 확인 차이 0 이면 destroy
rclone copy $B $RR -q --transfers 8
if rclone check $B $RR --one-way --size-only -q 2>/root/vr/check.txt; then
  say "[R2 확인 차이 0] -> destroy"
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함: $(tail -3 /root/vr/check.txt | tr '\n' ' ')"; echo "$(date) 차이 있음" | rclone rcat $MR/vres_CHECK_FAIL
fi
