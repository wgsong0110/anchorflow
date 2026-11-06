#!/bin/bash
# 절 CM17 묶음 astick (vast.ai A6000 1 장 인스턴스 onstart -> tmux astick; env R2_ACCESS_KEY R2_SECRET R2_ENDPOINT AF_COMMIT, vast 가 주는 CONTAINER_ID·CONTAINER_API_KEY):
#   (A) 가우스 점 고착 바닥 (절 CM8 a6 명령 그대로: --v=a, HHT ρ 0.8, K 2 x 뉴턴 2 x PCG 6, εN 1e-2 · εP 1e-1, 결정적) 의 고착 몫 법선 성분을 한쪽 구속으로 (사용자 승인 11-06 정의 변경):
#   판 a = 원래 (새 코드, 플래그 없음 -- 끄면 같은지 a6 해시로), u1 = (i) --fl_up 1 (접선 고착), u2 = (ii) --fl_up 2 (접선 자유). 씬 lego L000·0000·0013·0018.
#   모든 판 + --pen --hgt --pdiag --dump --dump_F --dump_part (진단·덤프, 풀이 불변). 렌더 (3DGS 알파블렌딩, exe/lb_render.py) 는 여기서, 합성은 로컬 (exe/fem_astick_compose_local.sh).
# 출력 R2 lbours/fem_cm/astick/<판>_<씬>, 표 lbours/fem_cm/marks/astick_table.md, 로그 marks/astick.log, 끝 표식 marks/astick.END.
# 준비 (pip·자료·코드) 를 다 끝낸 뒤 GPU 일을 이어서. 끝나면 rclone check 차이 0 일 때 스스로 destroy (-y), 아니면 그대로 두고 marks/astick_CHECK_FAIL.
set -u
C=${AF_COMMIT:?}
R=r2:storage/result/anchorflow; MR=$R/lbours/fem_cm/marks
mkdir -p /workspace/shm /workspace/tmpd /root/as
LOG=/root/as/astick.log
say() { echo "[astick $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MR/astick.log -q 2>/dev/null; }
fail() { say "$*"; echo "$* $(date)" | rclone rcat $MR/astick.END
  [ -d /dev/shm/dkta/lbours/fem_cm/astick ] && rclone copy /dev/shm/dkta/lbours/fem_cm/astick $R/lbours/fem_cm/astick -q
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
[ -n "$PL" ] && [ -n "$PD" ] && [ "$PL" -lt "$PD" ] && { echo "전력 제한 $PL < 기본 $PD" | rclone rcat $MR/astick_POWERCAP; fail "[전력 제한] power.limit $PL W < 기본 $PD W -> destroy"; }
# ---- 클러스터 경로 그대로 (exe/vast_nclamp.sh 와 같은 규약)
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
PYS=$(command -v python); ln -sf $PYS /home/dkta/.conda/envs/af/bin/python
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
if [ ! -f /root/as/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib vastai 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, h5py, matplotlib, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/as/pip.done
fi
W=/home/dkta/work; MI=$R/_vastmirror
SC="L000 0000 0013 0018"
rclone copyto $R/vastpkg/L000.pt $W/one/L000/L000.pt -q
for t in 0000 0013 0018; do rclone copyto $MI/lbpt/test/$t.pt $W/lbpt/test/$t.pt -q
  rclone copy $R/lbdata/test/$t $W/lbdata/test/$t -q --include meta.json --include sim_0000000000.h5; done
rclone copy $MI/i-physgaussian $W/i-physgaussian --transfers 16 -q
rclone copy $MI/pgmodel/lego_whitebg-trained $W/pgmodel/lego_whitebg-trained --transfers 16 -q
rclone copy $MI/lbdata $W/lbdata --transfers 16 -q
rclone copyto $MI/stab/lego/init/sim_0000000000.h5 $W/stab/lego/init/sim_0000000000.h5 -q
# ---- 코드 (github, 인스턴스에서 편집 안 함)
[ -d /root/af/.git ] || git clone -q https://github.com/wgsong0110/anchorflow.git /root/af
git -C /root/af fetch -q origin implicit-bspline; git -C /root/af checkout -q $C || fail "[코드 실패] $C"
D=/dev/shm/dkta/impwt/$C; [ -f $D/.done ] || { rm -rf $D; git clone -q /root/af $D && git -C $D checkout -q $C && echo $C > $D/.done; }
cd $D || exit 1
say "[코드] $(git log --oneline -1 | cut -c1-120)"
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_as XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
PY=/home/dkta/.conda/envs/af/bin/python
B=/dev/shm/dkta/lbours/fem_cm/astick; RR=$R/lbours/fem_cm/astick; M=/dev/shm/dkta/lbours/fem_cm/marks
mkdir -p $B $M $CUPY_CACHE_DIR
export CM_SCN=exe/fem_cm8_scn.sh
set +u
source exe/fem_cm8_scn.sh
for t in $SC; do cm_data $t; done
say "[자료] $(du -sh $W/* 2>&1 | tr '\n' ' ') lego 궤적 $(ls $LD/*.pt $SD/L000.pt 2>&1 | tr '\n' ' ') 정답 meta $(ls $W/lbdata/test/*/meta.json $W/lbdata/test_low/L000/meta.json 2>&1 | tr '\n' ' ')"
[ -f /root/as/pip.done ] && [ -s $SD/L000.pt ] && [ -s $LD/0000.pt ] && [ -s $LD/0013.pt ] && [ -s $LD/0018.pt ] && [ -f $W/lbdata/test/0000/meta.json ] && [ -f $W/lbdata/test/0013/meta.json ] && [ -f $W/lbdata/test/0018/meta.json ] \
  && [ -f $W/lbdata/test_low/L000/meta.json ] && [ -f $W/lbdata/config.json ] && [ -s $W/stab/lego/init/sim_0000000000.h5 ] && [ -d $W/pgmodel/lego_whitebg-trained ] \
  || fail "[세팅 실패] pip $(ls /root/as/pip.done 2>&1) lego $(ls $LD $SD 2>&1 | tr '\n' ' ') lbdata $(ls $W/lbdata $W/lbdata/test 2>&1 | tr '\n' ' ')"
say "[준비 끝] GPU 일 시작"
table() { $PY exe/fem_astick_table.py $B > $M/astick_table.md.tmp 2>$M/astick_table.err && mv $M/astick_table.md.tmp $M/astick_table.md; rclone copy $M/astick_table.md $MR -q; rclone copy $M/astick_table.err $MR -q; }
G="--ga hht --rho 0.8 --K 2 --newton 2"; T="--tolN 1e-2 --tolP 1e-1"
declare -A NV=([a]="" [u1]="--fl_up 1" [u2]="--fl_up 2")
DW="--det --v=a --pen --hgt --pdiag --dump --dump_F --dump_part"
run_one() { local V=$1 t=$2; shift 2
  local O=$B/${V}_$t; mkdir -p $O
  ( source exe/fem_cm8_scn.sh && cm_scene $t "$@" ${NV[$V]} && echo "카드 0 $C $(date '+%F %T') $SIM $DA $XA" > $O/run_info.txt \
    && CUDA_VISIBLE_DEVICES=0 timeout 2400 $PY -u $SIM --out $O --tag $t $DA $XA < /dev/null > $O/log.txt 2>&1; echo "rc=$? $(date '+%F %T')" >> $O/run_info.txt )
  say "  $V $t $(tail -1 $O/run_info.txt) $(grep -a '롤아웃\|rror\|발산' $O/log.txt | tail -2 | cut -c1-220 | tr '\n' ' ')"
  rclone copy $O $RR/${V}_$t -q
  table; }
# ---- 연기 (3 프레임): 새 커널 컴파일·실행 확인 (u1·u2 L000 결정적). 실패하면 기록 올리고 destroy
smk() { local V=$1 t=$2; local O=$B/smoke_${V}_$t; mkdir -p $O
  ( source exe/fem_cm8_scn.sh && cm_scene $t $G --cg 6 $T --det --v=a --pen --hgt ${NV[$V]} && echo "$SIM $DA $XA" > $O/cmd.txt && CUDA_VISIBLE_DEVICES=0 timeout 900 $PY -u $SIM --out $O --tag $t $DA $XA --frames 3 < /dev/null > $O/log.txt 2>&1; echo "rc=$?" > $O/rc.txt )
  grep -q '"fup"' $O/fem.jsonl 2>/dev/null && [ "$(cat $O/rc.txt)" = rc=0 ] && [ -f $O/summary.json ] && grep -q -- "--fl_up ${V#u}" $O/cmd.txt && grep -q -- '--floor vel' $O/cmd.txt && ! grep -q -- '--fl_bar\|--y0eb\|--fl_bst\|--fl_vres\|--fl_nclamp' $O/cmd.txt; }
smk u1 L000 && smk u2 L000 \
  && say "[연기 통과] $(for q in $B/smoke_*; do echo "$(basename $q) $(tail -1 $q/fem.jsonl | grep -o '"fup": {[^}]*}\|"err": [^,]*' | tr '\n' ' ')"; done | tr '\n' ' ')" \
  || fail "[연기 실패] $(for q in $B/smoke_*; do echo "$(basename $q) $(cat $q/rc.txt 2>/dev/null) $(grep -a 'rror\|Trace' $q/log.txt | tail -3 | cut -c1-300)"; done | tr '\n' ' ')"
say "[명령 확인] (i) L000: $(cat $B/smoke_u1_L000/cmd.txt)"
rclone copy $B $RR -q
lane() { local V=$1 t; shift; for t in "$@"; do run_one $V $t $G --cg 6 $T $DW; done; }       # 세 줄이 L000 (노드 상자 폭증 가능) 을 같은 때 돌리지 않게 씬 순서를 돌림
lane a L000 0000 0013 0018 & sleep 5; lane u1 0000 0013 0018 L000 & sleep 5; lane u2 0013 0018 L000 0000 & wait
# GPU 메모리 부족 (세 줄 동시) 으로 멈춘 판은 단독으로 다시 (노드 상자 폭증이면 단독도 같음 -- 기록으로 남김)
for O in $B/a_* $B/u1_* $B/u2_*; do
  if [ -f $O/summary.json ]; then grep -q '"oom"' $O/summary.json || continue; else grep -aq "OutOfMemory\|out of memory" $O/log.txt 2>/dev/null || continue; fi
  V=$(basename $O); t=${V#*_}; V=${V%%_*}
  say "[메모리 부족 -> 단독 다시] ${V}_$t ($(grep -o '"diverged": [0-9]*' $O/summary.json 2>/dev/null))"; mv $O $O.oom3; rclone copy $O.oom3 $RR/${V}_$t.oom3 -q
  run_one $V $t $G --cg 6 $T $DW; done
say "[판정 끝] (A) 해시 $(for t in 0000 0013 0018; do echo "$t $(grep -o '"x_sha1": "[0-9a-f]*"' $B/a_$t/summary.json)"; done | tr '\n' ' ') (a6 f444ba6042 · 48a7997a03 · 623d5873fd)"
table
# ---- 영상: 덤프 -> h5 -> 3DGS 렌더 (합성은 로컬)
rend1() { local V=$1 t=$2 O=$B/$1_$2 TD
  [ -f $O/pred.pt ] || { say "[영상 없음] ${V}_$t 덤프 없음"; return; }
  case $t in L000) TD=$W/lbdata/test_low/L000;; *) TD=$W/lbdata/test/$t;; esac
  $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $TD --out $O/h5 > $O/h5.log 2>&1
  mkdir -p $O/vj
  CUDA_VISIBLE_DEVICES=0 $PY -u exe/lb_render.py --sim $O/h5 --config $W/lbdata/config.json --shape lego --out $O/vj/fem_lego.mp4 --frame_label 0 --caption " " < /dev/null > $O/vj/r.log 2>&1
  say "  [렌더] ${V}_$t $(ls -la $O/vj/fem_lego.mp4 2>&1 | cut -c1-200) $(tail -1 $O/vj/r.log | cut -c1-200)"
  rclone copy $O $RR/${V}_$t -q --exclude pred.pt; }
for t in $SC; do rend1 a $t & sleep 3; rend1 u1 $t & sleep 3; rend1 u2 $t & wait; done
table
echo "[astick 끝] $(date '+%F %T')" > $M/astick.END; rclone copy $M/astick.END $MR -q
say "[astick 끝]"
# ---- 끝: R2 확인 차이 0 이면 destroy
rclone copy $B $RR -q --transfers 8
if rclone check $B $RR --one-way --size-only -q 2>/root/as/check.txt; then
  say "[R2 확인 차이 0] -> destroy"
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함: $(tail -3 /root/as/check.txt | tr '\n' ' ')"; echo "$(date) 차이 있음" | rclone rcat $MR/astick_CHECK_FAIL
fi
