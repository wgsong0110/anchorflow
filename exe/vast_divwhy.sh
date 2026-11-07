#!/bin/bash
# 절 S.10 묶음 divwhy (vast.ai A6000 2 장 인스턴스 onstart -> tmux divwhy; env R2_ACCESS_KEY R2_SECRET R2_ENDPOINT AF_COMMIT, vast 가 주는 CONTAINER_ID·CONTAINER_API_KEY):
#   exe/stab_divwhy.sh (lego L000 발산 판 10 개를 원래 명령 그대로 재현 + 진단 켬 판; 기존 기록 대조).
# 출력 R2 lbours/divwhy/ (판별 디렉토리, divwhy_table.md, divwhy.log, 끝 표식 divwhy.END). exe/vast_k4why.sh 와 같은 규약.
# 준비 (pip·자료·코드·기존 기록) 를 다 끝낸 뒤 GPU 일을 이어서. 끝나면 rclone check 차이 0 일 때 스스로 destroy (-y), 아니면 그대로 두고 divwhy_CHECK_FAIL.
set -u
C=${AF_COMMIT:?}
R=r2:storage/result/anchorflow; RR=$R/lbours/divwhy
mkdir -p /workspace/shm /workspace/tmpd /root/dw
LOG=/root/dw/vast.log
say() { echo "[divwhy $(date '+%F %T')] $*" | tee -a $LOG; command -v rclone >/dev/null && rclone copyto $LOG $RR/vast.log -q 2>/dev/null; }
B=/dev/shm/dkta/lbours/divwhy
fail() { say "$*"; echo "$* $(date)" | rclone rcat $RR/divwhy.END
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
for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
  PL=$(nvidia-smi -i $i --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1); PD=$(nvidia-smi -i $i --query-gpu=power.default_limit --format=csv,noheader,nounits | cut -d. -f1)
  say "[GPU $i] power.limit $PL W 기본 $PD W"
  [ -n "$PL" ] && [ -n "$PD" ] && [ "$PL" -lt "$PD" ] && { echo "카드 $i 전력 제한 $PL < 기본 $PD" | rclone rcat $RR/POWERCAP; fail "[전력 제한] 카드 $i power.limit $PL W < 기본 $PD W -> destroy"; }
done
NG=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l); [ "$NG" -ge 2 ] || fail "[카드 수] $NG < 2"
# ---- 클러스터 경로 그대로 (exe/vast_k4why.sh 와 같은 규약)
mkdir -p /home/dkta/work /home/dkta/.conda/envs/af/bin
[ -e /dev/shm/dkta ] || ln -s /workspace/shm /dev/shm/dkta
[ -e /tmp/dkta ] || ln -s /workspace/tmpd /tmp/dkta
PYS=$(command -v python); ln -sf $PYS /home/dkta/.conda/envs/af/bin/python
export HOME=/root PATH=/home/dkta/.conda/envs/af/bin:$PATH
if [ ! -f /root/dw/pip.done ]; then
  pip install -q "numpy<2" "warp-lang==0.10.1" "taichi==1.6.0" h5py scikit-image cupy-cuda12x tqdm plyfile scipy opencv-python-headless matplotlib vastai 2>&1 | tail -2 | tee -a $LOG
  python -c "import torch, cupy; print('[pip] torch', torch.__version__, torch.cuda.is_available(), 'cupy', cupy.__version__)" 2>&1 | tail -1 | tee -a $LOG && touch /root/dw/pip.done
fi
W=/home/dkta/work
rclone copyto $R/vastpkg/L000.pt $W/one/L000/L000.pt -q
# ---- 기존 기록 (명령·프레임별 오차)
mkdir -p $B/ref
for p in "k4 stab_k4why/k4o" "fa fem_cm/astick/a_L000" "fu1 fem_cm/astick/u1_L000" "fu2 fem_cm/astick/u2_L000" "sa fem_cm/astick_snh/sa_L000" "su1 fem_cm/astick_snh/su1_L000" "su2 fem_cm/astick_snh/su2_L000" \
         "vs fem_cm/vres/vs_L000" "vw fem_cm/vres/vw_L000" "nc fem_cm/nclamp/nc_L000"; do
  set -- $p; mkdir -p $B/ref/$1
  rclone copy $R/lbours/$2 $B/ref/$1 --include fem.jsonl --include summary.json --include run_info.txt --include cmd.txt -q
  [ -s $B/ref/$1/fem.jsonl ] || fail "[기존 기록 없음] $1 $2"
done
# ---- 코드 (github, 인스턴스에서 편집 안 함)
[ -d /root/af/.git ] || git clone -q https://github.com/wgsong0110/anchorflow.git /root/af
git -C /root/af fetch -q origin implicit-bspline; git -C /root/af checkout -q $C || fail "[코드 실패] $C"
D=/dev/shm/dkta/impwt/$C; [ -f $D/.done ] || { rm -rf $D; git clone -q /root/af $D && git -C $D checkout -q $C && echo $C > $D/.done; }
cd $D || exit 1
say "[코드] $(git log --oneline -1 | cut -c1-120)"
export PYTHONUTF8=1 OMP_NUM_THREADS=8 CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_dw XDG_CACHE_HOME=/dev/shm/dkta/cache
PY=/home/dkta/.conda/envs/af/bin/python
mkdir -p $B $CUPY_CACHE_DIR
set +u
source exe/fem_cm8_scn.sh
cm_data L000
[ -f /root/dw/pip.done ] && [ -s $SD/L000.pt ] || fail "[세팅 실패] pip $(ls /root/dw/pip.done 2>&1) L000 $(ls -la $SD 2>&1 | tr '\n' ' ')"
say "[준비 끝] GPU 일 시작 (L000 $(du -h $SD/L000.pt | cut -f1))"
# ---- 연기 (4 프레임): 판 다섯 (B K4·A·A(i) SNH·저항·클램프) 진단 켬·끔 프레임별 해시 같아야, 원자료에 힘 (n_F) 있어야
SM=$B/smoke; mkdir -p $SM
smk() { local r=$1 g=$2 m=$3 c add O=$SM/$1$3; mkdir -p $O
  c=$(head -1 $B/ref/$r/run_info.txt 2>/dev/null | sed 's/^.*exe\/fem_an_rt_sim.py/exe\/fem_an_rt_sim.py/')
  [ -n "$c" ] || c=$(head -1 $B/ref/$r/cmd.txt | sed 's/^.*exe\/fem_an_rt_sim.py/exe\/fem_an_rt_sim.py/; s/ --tag [^ ]*//; s/ --out [^ ]*//; s/ --fhash//')
  c=$(echo "$c" | sed 's/--frames 80/--frames 4/')
  if [ $m = o ]; then add="--fhash"; else add="--wdiag 1 --wdiag_end 1000 --pdiag --fhash"; fi
  echo "$PY -u $c --tag L000 --out $O $add" > $O/cmd.txt
  CUDA_VISIBLE_DEVICES=$g timeout 1500 $PY -u $c --tag L000 --out $O $add < /dev/null > $O/log.txt 2>&1; echo "rc=$?" > $O/rc.txt; }
( for r in k4 fa nc; do smk $r 0 o; smk $r 0 w; done ) &
( for r in su1 vs; do smk $r 1 o; smk $r 1 w; done ) &
wait
OK=1; MSG=""
for r in k4 fa nc su1 vs; do
  HO=$(grep -o '"xh": "[0-9a-f]*"' $SM/${r}o/fem.jsonl 2>/dev/null | tr '\n' ' '); HN=$(grep -o '"xh": "[0-9a-f]*"' $SM/${r}w/fem.jsonl 2>/dev/null | tr '\n' ' ')
  NF=$($PY -c "import numpy as np; z=np.load('$SM/${r}w/wd_f2.npz'); print(int('n_F' in z and 'n_Bmax' in z and z['n_F'].shape[0] > 0))" 2>/dev/null)
  EO=$(head -4 $B/ref/$r/fem.jsonl | grep -o '"err": [0-9.e-]*' | tr '\n' ' '); EN=$(grep -o '"err": [0-9.e-]*' $SM/${r}o/fem.jsonl 2>/dev/null | tr '\n' ' ')
  MSG="$MSG | $r rc $(cat $SM/${r}o/rc.txt) $(cat $SM/${r}w/rc.txt) 해시 $([ -n "$HO" ] && [ "$HO" = "$HN" ] && echo 같음 || echo 다름) 오차 기존 $([ "$EO" = "$EN" ] && echo 같음 || echo 다름) n_F $NF"
  { [ "$(cat $SM/${r}o/rc.txt)" = rc=0 ] && [ "$(cat $SM/${r}w/rc.txt)" = rc=0 ] && [ -n "$HO" ] && [ "$HO" = "$HN" ] && [ "$NF" = 1 ]; } || OK=0
done
rclone copy $B $RR -q
[ $OK = 1 ] && say "[연기 통과] $MSG" || fail "[연기 실패] $MSG $(grep -ah 'rror\|Trace' $SM/*/log.txt | tail -4 | cut -c1-400 | tr '\n' ' ')"
# ---- 본 묶음 (업로드 10 분마다)
( while [ ! -f $B/divwhy.END ]; do sleep 600; rclone copy $B $RR -q; done ) &
bash exe/stab_divwhy.sh $B
say "[본 묶음 끝] $(tail -24 $B/divwhy.log | tr '\n' ' ')"
rclone copy $B $RR -q --transfers 8
if rclone check $B $RR --one-way --size-only -q 2>/root/dw/check.txt; then
  say "[R2 확인 차이 0] -> destroy"
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x} 2>&1 | tee -a $LOG
else
  say "[R2 확인 차이 있음] destroy 안 함: $(tail -3 /root/dw/check.txt | tr '\n' ' ')"; echo "$(date) 차이 있음" | rclone rcat $RR/divwhy_CHECK_FAIL
fi
