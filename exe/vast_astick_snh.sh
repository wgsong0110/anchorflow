#!/bin/bash
# 절 CM18 묶음 astick_snh (vast.ai A6000 1 장 인스턴스 onstart -> tmux asnh; env R2_ACCESS_KEY R2_SECRET R2_ENDPOINT AF_COMMIT, vast 가 주는 CONTAINER_ID·CONTAINER_API_KEY):
#   절 CM17 의 A 바닥 세 판 ((A) 원래·(i) --fl_up 1·(ii) --fl_up 2) 을 lego L000 에서 탄성 Stable Neo-Hookean (--elx snh, 절 CM13 (2) 와 같은 식, μ_s = μ, λ_s = λ + μ) 으로 (사용자 승인 11-06 정의 변경).
#   판 sa·su1·su2 = SNH, fa·fu1·fu2 = 절 CM17 FCR 판 같은 명령 + 진단 --qjd 만 (가우스 점 J 측정용, 절 CM17 결과와 프레임별 같은지 = 끄면 같음 대조). 모두 결정적, 카드 0 에 하나씩 차례로 (메모리 부족 섞임 없게).
#   모든 판 = 절 CM17 명령 (--v=a, HHT ρ 0.8, K 2 x 뉴턴 2 x PCG 6, εN 1e-2 · εP 1e-1, --det --pen --hgt --pdiag --dump --dump_F --dump_part) + --qjd. 렌더 (3DGS, exe/lb_render.py) 는 SNH 세 판만 여기서, 합성은 로컬 (exe/fem_astick_snh_compose_local.sh).
# 출력 R2 lbours/fem_cm/astick_snh/<판>_L000, 표 lbours/fem_cm/marks/astick_snh_table.md, 로그 marks/astick_snh.log, 끝 표식 marks/astick_snh.END.
# 준비 (pip·자료·코드) 를 다 끝낸 뒤 GPU 일을 이어서. 끝나면 rclone check 차이 0 일 때 스스로 destroy (-y), 아니면 그대로 두고 marks/astick_snh_CHECK_FAIL.
set -u
C=${AF_COMMIT:?}
R=r2:storage/result/anchorflow; MR=$R/lbours/fem_cm/marks
mkdir -p /workspace/shm /workspace/tmpd /root/asn
LOG=/root/asn/astick_snh.log
say() { echo "[astick_snh $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $MR/astick_snh.log -q 2>/dev/null; }
fail() { say "$*"; echo "$* $(date)" | rclone rcat $MR/astick_snh.END
  [ -d /dev/shm/dkta/lbours/fem_cm/astick_snh ] && rclone copy /dev/shm/dkta/lbours/fem_cm/astick_snh $R/lbours/fem_cm/astick_snh -q
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
[ -n "$PL" ] && [ -n "$PD" ] && [ "$PL" -lt "$PD" ] && { echo "전력 제한 $PL < 기본 $PD" | rclone rcat $MR/astick_snh_POWERCAP; fail "[전력 제한] power.limit $PL W < 기본 $PD W -> destroy"; }
# ---- 클러스터 경로 그대로 (exe/vast_nclamp.sh 와 같은 규약)
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin /home/dkta/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
PYS=$(command -v python); ln -sf $PYS /home/dkta/.conda/envs/af/bin/python
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
if [ ! -f /root/asn/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image lpips cupy-cuda12x tqdm plyfile scipy imageio imageio-ffmpeg av opencv-python-headless matplotlib vastai 2>&1 | tail -2 | tee -a $LOG
  cd /tmp && for f in diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl; do
    curl -sL -o $f "https://github.com/wgsong0110/anchorflow/releases/download/cuda-3dgs-torch251/$f"; done
  pip install -q --force-reinstall --no-deps /tmp/diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl /tmp/simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy, h5py, matplotlib, diff_gaussian_rasterization; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/asn/pip.done
fi
W=/home/dkta/work; MI=$R/_vastmirror
SC="L000"
rclone copyto $R/vastpkg/L000.pt $W/one/L000/L000.pt -q
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
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_asn XDG_CACHE_HOME=/dev/shm/dkta/cache TORCH_EXTENSIONS_DIR=/dev/shm/dkta/cache/torch_ext
PY=/home/dkta/.conda/envs/af/bin/python
B=/dev/shm/dkta/lbours/fem_cm/astick_snh; RR=$R/lbours/fem_cm/astick_snh; M=/dev/shm/dkta/lbours/fem_cm/marks
mkdir -p $B $M $CUPY_CACHE_DIR
export CM_SCN=exe/fem_cm8_scn.sh
set +u
source exe/fem_cm8_scn.sh
for t in $SC; do cm_data $t; done
say "[자료] $(du -sh $W/* 2>&1 | tr '\n' ' ') lego 궤적 $(ls $LD/*.pt $SD/L000.pt 2>&1 | tr '\n' ' ') 정답 meta $(ls $W/lbdata/test/*/meta.json $W/lbdata/test_low/L000/meta.json 2>&1 | tr '\n' ' ')"
[ -f /root/asn/pip.done ] && [ -s $SD/L000.pt ] && [ -f $W/lbdata/test_low/L000/meta.json ] && [ -f $W/lbdata/config.json ] && [ -s $W/stab/lego/init/sim_0000000000.h5 ] && [ -d $W/pgmodel/lego_whitebg-trained ] \
  || fail "[세팅 실패] pip $(ls /root/asn/pip.done 2>&1) lego $(ls $LD $SD 2>&1 | tr '\n' ' ') lbdata $(ls $W/lbdata $W/lbdata/test 2>&1 | tr '\n' ' ')"
say "[준비 끝] GPU 일 시작"
mkdir -p $B/cm17
for v in a u1 u2; do rclone copy $R/lbours/fem_cm/astick/${v}_L000 $B/cm17/${v}_L000 -q --include summary.json --include fem.jsonl --include run_info.txt; done
say "[절 CM17 결과] $(for v in a u1 u2; do echo "$v $(grep -o '"diverged": [0-9a-z]*' $B/cm17/${v}_L000/summary.json) $(wc -l < $B/cm17/${v}_L000/fem.jsonl)"; done | tr '\n' ' ')"
table() { $PY exe/fem_astick_snh_table.py $B > $M/astick_snh_table.md.tmp 2>$M/astick_snh_table.err && mv $M/astick_snh_table.md.tmp $M/astick_snh_table.md; rclone copy $M/astick_snh_table.md $MR -q; rclone copy $M/astick_snh_table.err $MR -q; }
G="--ga hht --rho 0.8 --K 2 --newton 2"; T="--tolN 1e-2 --tolP 1e-1"
declare -A NV=([fa]="" [fu1]="--fl_up 1" [fu2]="--fl_up 2" [sa]="--elx snh" [su1]="--fl_up 1 --elx snh" [su2]="--fl_up 2 --elx snh")
DW="--det --v=a --pen --hgt --pdiag --dump --dump_F --dump_part --qjd"
run_one() { local V=$1 t=$2; shift 2
  local O=$B/${V}_$t; mkdir -p $O
  ( source exe/fem_cm8_scn.sh && cm_scene $t "$@" ${NV[$V]} && echo "카드 0 $C $(date '+%F %T') $SIM $DA $XA" > $O/run_info.txt \
    && CUDA_VISIBLE_DEVICES=0 timeout 2400 $PY -u $SIM --out $O --tag $t $DA $XA < /dev/null > $O/log.txt 2>&1; echo "rc=$? $(date '+%F %T')" >> $O/run_info.txt )
  say "  $V $t $(tail -1 $O/run_info.txt) $(grep -a '롤아웃\|rror\|발산' $O/log.txt | tail -2 | cut -c1-220 | tr '\n' ' ')"
  rclone copy $O $RR/${V}_$t -q
  table; }
# ---- 연기 (3 프레임): SNH 커널 컴파일·실행 + 진단 확인 (sa·su1). 실패하면 기록 올리고 destroy
smk() { local V=$1 t=$2; local O=$B/smoke_${V}_$t; mkdir -p $O
  ( source exe/fem_cm8_scn.sh && cm_scene $t $G --cg 6 $T --det --v=a --pen --hgt --pdiag --qjd ${NV[$V]} && echo "$SIM $DA $XA" > $O/cmd.txt && CUDA_VISIBLE_DEVICES=0 timeout 900 $PY -u $SIM --out $O --tag $t $DA $XA --frames 3 < /dev/null > $O/log.txt 2>&1; echo "rc=$?" > $O/rc.txt )
  grep -q '"qj"' $O/fem.jsonl 2>/dev/null && [ "$(cat $O/rc.txt)" = rc=0 ] && [ -f $O/summary.json ] && grep -q -- '"elx": "snh"' $O/summary.json && grep -q -- '--floor vel' $O/cmd.txt && ! grep -q -- '--fl_bar\|--y0eb\|--fl_bst\|--fl_vres\|--fl_nclamp' $O/cmd.txt; }
smk sa L000 && smk su1 L000 \
  && say "[연기 통과] $(for q in $B/smoke_*; do echo "$(basename $q) $(tail -1 $q/fem.jsonl | grep -o '"err": [^,]*\|"qj": {[^}]*}' | tr '\n' ' ')"; done | tr '\n' ' ')" \
  || fail "[연기 실패] $(for q in $B/smoke_*; do echo "$(basename $q) $(cat $q/rc.txt 2>/dev/null) $(grep -a 'rror\|Trace' $q/log.txt | tail -3 | cut -c1-300)"; done | tr '\n' ' ')"
say "[명령 확인] (i)+SNH L000: $(cat $B/smoke_su1_L000/cmd.txt)"
rclone copy $B $RR -q
for V in sa su1 su2 fa fu1 fu2; do run_one $V L000 $G --cg 6 $T $DW; done
say "[판정 끝] $(for V in sa su1 su2 fa fu1 fu2; do echo "$V $(grep -o '"diverged": [0-9a-z]*\|"err_mean_done": [0-9.e-]*' $B/${V}_L000/summary.json | tr '\n' ' ')"; done | tr '\n' ' ')"
table
# ---- 영상: SNH 세 판 덤프 -> h5 -> 3DGS 렌더 (합성은 로컬)
rend1() { local V=$1 t=$2 O=$B/$1_$2 TD=$W/lbdata/test_low/L000
  [ -f $O/pred.pt ] || { say "[영상 없음] ${V}_$t 덤프 없음"; return; }
  $PY -u exe/lb_roll_to_h5.py --dump $O/pred.pt --traj $TD --out $O/h5 > $O/h5.log 2>&1
  mkdir -p $O/vj
  CUDA_VISIBLE_DEVICES=0 $PY -u exe/lb_render.py --sim $O/h5 --config $W/lbdata/config.json --shape lego --out $O/vj/fem_lego.mp4 --frame_label 0 --caption " " < /dev/null > $O/vj/r.log 2>&1
  say "  [렌더] ${V}_$t $(ls -la $O/vj/fem_lego.mp4 2>&1 | cut -c1-200) $(tail -1 $O/vj/r.log | cut -c1-200)"
  rclone copy $O $RR/${V}_$t -q --exclude pred.pt; }
rend1 sa L000 & sleep 3; rend1 su1 L000 & sleep 3; rend1 su2 L000 & wait
table
echo "[astick_snh 끝] $(date '+%F %T')" > $M/astick_snh.END; rclone copy $M/astick_snh.END $MR -q
say "[astick_snh 끝]"
# ---- 끝: R2 확인 차이 0 이면 destroy
rclone copy $B $RR -q --transfers 8
if rclone check $B $RR --one-way --size-only -q 2>/root/asn/check.txt; then
  say "[R2 확인 차이 0] -> destroy"
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함: $(tail -3 /root/asn/check.txt | tr '\n' ' ')"; echo "$(date) 차이 있음" | rclone rcat $MR/astick_snh_CHECK_FAIL
fi
