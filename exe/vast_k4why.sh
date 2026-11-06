#!/bin/bash
# 절 S.9 묶음 k4why (vast.ai A6000 1 장 인스턴스 onstart -> tmux k4why; env R2_ACCESS_KEY R2_SECRET R2_ENDPOINT AF_COMMIT, vast 가 주는 CONTAINER_ID·CONTAINER_API_KEY):
#   exe/stab_k4why.sh (우리 (B) L000 K 4 발산 -- 몸통 안 가우스 점 뒤집힘의 원인 측정; 진단 켬·끔 대조).
# 출력 R2 lbours/stab_k4why/ (판별 디렉토리, k4why_table.md, k4why.log, 끝 표식 k4why.END).
# 준비 (pip·자료·코드) 를 다 끝낸 뒤 GPU 일을 이어서. 끝나면 rclone check 차이 0 일 때 스스로 destroy (-y), 아니면 그대로 두고 k4why_CHECK_FAIL.
set -u
C=${AF_COMMIT:?}
R=r2:storage/result/anchorflow; RR=$R/lbours/stab_k4why
mkdir -p /workspace/shm /workspace/tmpd /root/kw
LOG=/root/kw/vast.log
say() { echo "[k4why $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $RR/vast.log -q 2>/dev/null; }
B=/dev/shm/dkta/lbours/stab_k4why
fail() { say "$*"; echo "$* $(date)" | rclone rcat $RR/k4why.END
  [ -d $B ] && rclone copy $B $RR -q
  say "[실패 -> destroy] 자료는 R2 에"; vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG; exit 1; }
if ! command -v rclone >/dev/null; then
  (apt-get update -qq && apt-get install -y -qq curl unzip git tmux) >/dev/null 2>&1
  curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/
fi
mkdir -p ~/.config/rclone
printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' "$R2_ACCESS_KEY" "$R2_SECRET" "$R2_ENDPOINT" > ~/.config/rclone/rclone.conf
say "[시작] 커밋 $C $(hostname) 인스턴스 ${CONTAINER_ID:-?}"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit,power.draw,clocks.max.sm,memory.total --format=csv | tee -a $LOG
PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | cut -d. -f1); PD=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
say "[GPU] power.limit $PL W 기본 $PD W"
[ -n "$PL" ] && [ -n "$PD" ] && [ "$PL" -lt "$PD" ] && { echo "전력 제한 $PL < 기본 $PD" | rclone rcat $RR/POWERCAP; fail "[전력 제한] power.limit $PL W < 기본 $PD W -> destroy"; }
# ---- 클러스터 경로 그대로 (exe/vast_astick.sh 와 같은 규약)
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
PYS=$(command -v python); ln -sf $PYS /home/dkta/.conda/envs/af/bin/python
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
if [ ! -f /root/kw/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image cupy-cuda12x tqdm plyfile scipy opencv-python-headless matplotlib vastai 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/kw/pip.done
fi
W=/home/dkta/work
rclone copyto $R/vastpkg/L000.pt $W/one/L000/L000.pt -q
# ---- 코드 (github, 인스턴스에서 편집 안 함)
[ -d /root/af/.git ] || git clone -q https://github.com/wgsong0110/anchorflow.git /root/af
git -C /root/af fetch -q origin implicit-bspline; git -C /root/af checkout -q $C || fail "[코드 실패] $C"
D=/dev/shm/dkta/impwt/$C; [ -f $D/.done ] || { rm -rf $D; git clone -q /root/af $D && git -C $D checkout -q $C && echo $C > $D/.done; }
cd $D || exit 1
say "[코드] $(git log --oneline -1 | cut -c1-120)"
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_kw XDG_CACHE_HOME=/dev/shm/dkta/cache
PY=/home/dkta/.conda/envs/af/bin/python
mkdir -p $B $CUPY_CACHE_DIR
set +u
source exe/fem_cm8_scn.sh
cm_data L000
[ -f /root/kw/pip.done ] && [ -s $SD/L000.pt ] || fail "[세팅 실패] pip $(ls /root/kw/pip.done 2>&1) L000 $(ls -la $SD 2>&1 | tr '\n' ' ')"
say "[준비 끝] GPU 일 시작 (L000 $(du -h $SD/L000.pt | cut -f1))"
# ---- 연기 (8 프레임 K 4): 진단 켬·끔 프레임별 해시 같아야, wd 기록 있어야
smk() { local n=$1; shift; local O=$B/smoke_$n; mkdir -p $O
  ( source exe/fem_cm8_scn.sh && cm_scene L000 --v=b --ga hht --rho 0.8 --K 4 --newton 2 --cg 4 --tolN 1e-2 --tolP 1e-1 --det && echo "$SIM $DA $XA $*" > $O/cmd.txt \
    && CUDA_VISIBLE_DEVICES=0 timeout 1200 $PY -u $SIM --out $O --tag L000 $DA $XA --frames 8 "$@" < /dev/null > $O/log.txt 2>&1; echo "rc=$?" > $O/rc.txt ); }
smk off --fhash & smk on --wdiag 1 --wdiag_end 2 --gfin --diag --pdiag --fhash --dump --dump_part & wait
HO=$(grep -o '"xh": "[0-9a-f]*"' $B/smoke_off/fem.jsonl 2>/dev/null | tr '\n' ' '); HN=$(grep -o '"xh": "[0-9a-f]*"' $B/smoke_on/fem.jsonl 2>/dev/null | tr '\n' ' ')
rclone copy $B $RR -q
if [ "$(cat $B/smoke_on/rc.txt)" = rc=0 ] && [ "$(cat $B/smoke_off/rc.txt)" = rc=0 ] && [ -n "$HO" ] && [ "$HO" = "$HN" ] && grep -q '"wd"' $B/smoke_on/fem.jsonl && [ -f $B/smoke_on/wd_f1.npz ]; then
  say "[연기 통과] 8 프레임 해시 같음: $HO"
else
  fail "[연기 실패] on $(cat $B/smoke_on/rc.txt 2>/dev/null) off $(cat $B/smoke_off/rc.txt 2>/dev/null) 해시 끔 [$HO] 켬 [$HN] $(grep -a 'rror\|Trace' $B/smoke_on/log.txt | tail -4 | cut -c1-400 | tr '\n' ' ')"
fi
# ---- 본 묶음 (업로드 10 분마다)
( while [ ! -f $B/k4why.END ]; do sleep 600; rclone copy $B $RR -q; done ) &
bash exe/stab_k4why.sh $B
say "[본 묶음 끝] $(tail -9 $B/k4why.log | tr '\n' ' ')"
rclone copy $B $RR -q --transfers 8
if rclone check $B $RR --one-way --size-only -q 2>/root/kw/check.txt; then
  say "[R2 확인 차이 0] -> destroy"
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함: $(tail -3 /root/kw/check.txt | tr '\n' ' ')"; echo "$(date) 차이 있음" | rclone rcat $RR/k4why_CHECK_FAIL
fi
