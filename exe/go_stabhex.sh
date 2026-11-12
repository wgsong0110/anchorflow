#!/bin/bash
# 서브스텝 안정성 '우리(육면체)' 묶음 stabhex 진입점 (11-12 지시; 틀은 exe/go_cellh.sh). 코드 디렉토리에서. 사용: bash exe/go_stabhex.sh <커밋> <단계>
#   data  = 세 형상 입력 .pt (exe/stabhex_data.py; 안정성 실험 0 프레임 입자·config 그대로)
#   chk   = 판 목록 (exe/stabhex_jobs.txt) 의 명령줄만 만들어 R2 에 (실행 안 함)
#   s0k   = 연기: 레고 N1·N2·N4 3 프레임 (npz 프레임 3 확인)
#   run   = 판 목록 전부 (이미 끝난 판 건너뜀) 를 빈 카드에 큐로 (기준 궤적 판 = 큰 N 먼저)
#   an    = 분석 (CPU, exe/stabhex_an.py) -> stabhex_an.json
#   vid   = 덤프 판 3DGS 렌더 (exe/rep_ip.py --render_npz, 같은 렌더러) + 비교 합성 (exe/stabhex_compose.py)
#   speed = 단독 시간 (카드 하나에 다른 계산 프로세스 0 일 때 차례로; 기록·덤프·진단 환경 변수 없음)
# 노드 / ·/tmp 는 안 씀: 코드·TMPDIR·캐시·입력 /dev/shm/dkta, 결과 /dev/shm/dkta/stabhex/work (/home 여유 21 GB 라; 판마다 R2 로).
export PATH=/home/dkta/bin:$PATH
H=$1; STG=$2
if [[ "$STG" == *,* ]]; then for st in ${STG//,/ }; do bash "$0" "$H" "$st"; done; exit 0; fi       # 여러 단계 차례로
R=r2:storage/result/anchorflow/lbours/stabhex
O=/dev/shm/dkta/stabhex/work; C=$(pwd -P); SD=/dev/shm/dkta/stabhex/data
W=/home/dkta/work/stab
export TMPDIR=/dev/shm/dkta/tmp; mkdir -p $TMPDIR $O $SD /dev/shm/dkta/cache
export PYTHONUTF8=1 OMP_NUM_THREADS=8 XDG_CACHE_HOME=/dev/shm/dkta/cache CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_stabhex
PY=/home/dkta/.conda/envs/af/bin/python
L=$O/go_$STG.log; say() { echo "[go $(date '+%F %T')] $*" >> $L; rclone copyto $L $R/go_$STG.log -q; }
say "시작 $(hostname) 커밋 $H 단계 $STG; /dev/shm $(df -h /dev/shm | tail -1 | awk '{print $4}') /home $(df -h /home | tail -1 | awk '{print $4}') 남음; 계산 프로세스 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
BASE=$(cat exe/stabhex_base.txt)
mkcmd() {  # 이름 형상 N 덤프(0|1) 프레임 기록(1|0) 환경 출력폴더 -> 명령줄 (stdout)
  local n=$1 s=$2 k=$3 dp=$4 fr=$5 st=$6 env=$7 out=$8 du=""
  [ "$dp" = 1 ] && du="--dump --dump_F "
  echo "$BASE" | sed -e "s#@ENV@#$env#" -e "s#@OUT@#$out#g" -e "s#@ST@#$st#" -e "s#@DATA@#$SD/$s.pt#" -e "s#@FR@#$fr#" -e "s#@TAG@#$n#" -e "s#@N@#$k#" -e "s#@DUMP@#$du#" -e "s#@S@#$s#g"
}
CL=/dev/shm/dkta/anerr_claim; mkdir -p $CL
NG=$(nvidia-smi -L | wc -l)
claim() { local me=$1 g f p
  for g in $(seq 0 $((NG - 1))); do
    f=$CL/$g
    if [ -f $f ]; then p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && continue; rm -f $f; fi
    [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" = 0 ] || continue
    ( set -o noclobber; echo $me > $f ) 2>/dev/null && { echo $g; return 0; }
  done; return 1; }
claim_wait() { local g; while :; do g=$(claim $1) && { echo $g; return; }; sleep 30; done; }
runone() {  # 이름 카드 (O/<이름>/cmd.txt 가 있어야 함; 카드 표식은 부른 쪽이 잡아 둠 -> 여기서 내 PID 로 바꿈) -- 실행, 끝나면 표식 풀고 R2
  local n=$1 g=$2; echo $BASHPID > $CL/$g; say "$n 카드 $g 시작"
  env CUDA_VISIBLE_DEVICES=$g timeout 14400 bash -c "$(cat $O/$n/cmd.txt)" < /dev/null > $O/$n/log.txt 2>&1
  local rc=$?; rm -f $CL/$g
  say "$n rc=$rc $(grep -ah '롤아웃\]\|\[stabhex\] 저장\|발산\]' $O/$n/log.txt | tail -2 | cut -c1-220 | tr '\n' ' ') $(grep -a 'rror' $O/$n/log.txt | tail -2 | cut -c1-300 | tr '\n' ' ')"
  [ -s $O/$n/res.npz ] && echo "rc=$rc $(date '+%F %T')" > $O/$n/DONE
  rclone copy $O/$n $R/$n -q --exclude "pred.pt" --exclude "*.tmp.npz" --transfers 8
  [ -f $O/$n/pred.pt ] && rclone copy $O/$n $R/$n -q --include "pred.pt"
  say "$n R2 올림"; }
jobs_of() { grep -v '^#' exe/stabhex_jobs.txt | awk 'NF>=5'; }
case $STG in
data) for s in lego mic ficus; do mkdir -p /dev/shm/dkta/stabhex/rr/$s      # 루프 안 렌더 (결과 불변; H9 의 X2·X3g 가 그 자리 speed4 에 묶여 있음) 용 meta: 0 프레임 h5·R = I
       echo "{\"h5\": \"$W/$s/init/sim_0000000000.h5\", \"R\": [[1,0,0],[0,1,0],[0,0,1]]}" > /dev/shm/dkta/stabhex/rr/$s/meta.json
       $PY -u exe/stabhex_data.py $W/$s $SD/$s.pt 60 >> $L 2>&1; say "$s 입력 rc=$? $(tail -1 $L | cut -c1-200)"; done;;
chk) jobs_of | while read -r n s k rep dp; do mkdir -p $O/$n; mkcmd $n $s $k $dp 60 1 "AF_E1_DG=1 AF_G1_DG=1 " $O/$n > $O/$n/cmd.txt
       rclone copyto $O/$n/cmd.txt $R/$n/cmd.txt -q; done
     say "판 $(jobs_of | wc -l) 개 명령줄; 첫 판: $(head -c 3000 $O/$(jobs_of | head -1 | awk '{print $1}')/cmd.txt)";;
s0k) PIDS=""; for k in 1 2 4; do n=smoke_N$k; rm -rf $O/$n; mkdir -p $O/$n; mkcmd $n lego $k 1 3 1 "AF_E1_DG=1 AF_G1_DG=1 " $O/$n > $O/$n/cmd.txt
       g=$(claim_wait $$); runone $n $g & PIDS="$PIDS $!"; sleep 20; done; wait $PIDS
     for k in 1 2 4; do n=smoke_N$k; say "$n 확인: $($PY -c "import numpy as np; z=np.load('$O/$n/res.npz'); print('프레임', int(z['frames_done']), 'x', z['x'].shape, 'nan', float(np.nanmax(z['stats'][:, 2])), 'it', z['it'].tolist(), 'K', int(z['substeps']), z['stop'])" 2>&1 | tail -1 | cut -c1-400)"; done;;
run) for n in $(jobs_of | awk '{print $1}'); do [ -s $O/$n/cmd.txt ] || { say "$n cmd.txt 없음 -- chk 먼저"; exit 1; }; done
     rm -f $O/go_run.END
     ( while [ ! -f $O/go_run.END ]; do sleep 300; ls $O/*/DONE 2>/dev/null | wc -l | xargs -I{} echo "끝난 판 {} / $(jobs_of | wc -l) $(date '+%F %T')" | rclone rcat $R/progress.txt
       nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>&1 | rclone rcat $R/apps_now.txt; done ) &
     PIDS=""
     for n in $(jobs_of | awk '{print $1}'); do
       [ -f $O/$n/DONE ] && continue
       g=$(claim_wait $$)                                                              # 빈 카드 (계산 프로세스 0·표식 없음) 생기면 바로 -- 표식 주인은 먼저 이 고리, 판이 뜨면 판 PID
       runone $n $g & PIDS="$PIDS $!"; sleep 25
     done
     wait $PIDS; touch $O/go_run.END;;
an)  $PY -u exe/stabhex_an.py --root $W --hex $O --jobs exe/stabhex_jobs.txt --out $O/stabhex_an.json > $O/stabhex_an.log 2>&1
     say "an rc=$? $(tail -3 $O/stabhex_an.log | cut -c1-300 | tr '\n' ' ')"
     rclone copy $O $R --include "stabhex_an.*" -q;;
vid) V=$O/vid; mkdir -p $V
     for n in $(jobs_of | awk '$5==1{print $1}'); do
       s=$(jobs_of | awk -v n=$n '$1==n{print $2}')
       [ -f $V/$n.mp4 ] && continue
       [ -f $O/$n/pred.pt ] || { say "$n pred.pt 없음 (발산?)"; continue; }
       g=$(claim_wait $$)
       $PY - <<PYEOF >> $V/render.log 2>&1
import torch, numpy as np
q = torch.load('$O/$n/pred.pt', map_location='cpu', weights_only=False)
X = q['pred'].float().numpy(); Fm = q['F'].float().numpy()                     # 프레임 1..60 (rep_ip 이 0 프레임을 먼저 그림)
np.savez('$V/${n}_full.npz', traj_full=X, F_full=Fm)
print('$n', X.shape, Fm.shape)
PYEOF
       (cd $C && CUDA_VISIBLE_DEVICES=$g $PY -u exe/rep_ip.py --sim $W/$s/init --cfg $W/$s/config.json --shape $s --method ours --frames 60 --no_ref \
         --render_npz $V/${n}_full.npz --video $V/${n}_raw.mp4 --out $V/${n}.r.npz) >> $V/render.log 2>&1
       rc=$?; rm -f $CL/$g $V/${n}_full.npz
       F=$($PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
       $F -y -loglevel error -i $V/${n}_raw.mp4 -vf "scale='min(2400,iw)':-2" -pix_fmt yuv420p -movflags +faststart -c:v libx264 -crf 18 $V/$n.mp4
       say "$n 렌더 rc=$rc $(ls -la $V/$n.mp4 2>&1 | cut -c1-200)"
     done
     OV=$V/oldvid; mkdir -p $OV; rclone copy r2:storage/result/anchorflow/stab_vid $OV -q --include "*.mp4"      # 이미 있는 영상 (다시 안 돌림)
     rclone copyto $R/code/NotoSansCJKkr-Regular.otf /dev/shm/dkta/stabhex/font.otf -q
     $PY -u exe/stabhex_compose.py --hexvid $V --old $OV --font /dev/shm/dkta/stabhex/font.otf --out $V/cmp >> $V/compose.log 2>&1; say "합성 rc=$? $(tail -2 $V/compose.log | tr '\n' ' ' | cut -c1-300)"
     rclone copy $V $R/vid -q --exclude "*_raw.mp4" --exclude "*.r.npz" --exclude "oldvid/**" --exclude "*_full.npz";;
speed) SP=$O/speed; mkdir -p $SP
     g=$(claim_wait $$); say "단독 카드 $g"
     for s in lego mic ficus; do for k in 1 2 4 8 16 32 64; do n=t_${s}_N$k; mkdir -p $SP/$n
       mkcmd $n $s $k 0 60 0 "" $SP/$n > $SP/$n/cmd.txt
       nvidia-smi -i $g --query-compute-apps=pid,used_memory --format=csv,noheader > $SP/$n/apps_at_start.txt 2>&1
       [ -s $SP/$n/apps_at_start.txt ] && { say "$n 카드 $g 에 다른 프로세스 -- 기다림"; while [ -n "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done; nvidia-smi -i $g --query-compute-apps=pid,used_memory --format=csv,noheader > $SP/$n/apps_at_start.txt; }
       env CUDA_VISIBLE_DEVICES=$g timeout 7200 bash -c "$(cat $SP/$n/cmd.txt)" < /dev/null > $SP/$n/log.txt 2>&1
       say "$n rc=$? $(grep -ah '롤아웃\]' $SP/$n/log.txt | tail -1 | cut -c1-200)"
     done; done
     rm -f $CL/$g
     rclone copy $SP $R/speed -q;;
esac
echo "끝 $(date '+%F %T')" > $O/go_$STG.END; rclone copyto $O/go_$STG.END $R/go_$STG.END -q
say "끝"
