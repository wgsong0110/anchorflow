#!/bin/bash
# 절 AM: 정답 MPM (lego L000, PG warp 솔버 정답 설정 그대로) 가속 판들을 차례로 돌리고 결과 불변을 비교한다.
#   AM_TIMED=1 : 단독 측정 -- 카드 1~6 이 모두 10 MiB 이하가 될 때까지 기다린 뒤 /home/dkta/work/gpu0.lock 을 잡고 GPU0 (판마다 다시 확인, 시작·끝 nvidia-smi 기록)
#   기본 (비측정, NOT_TIMED 표식): gpu0.lock 을 최대 30 분 기다려 GPU0 (다른 갈래 단독 측정과 직렬화), 못 잡으면 카드 1~6 중 메모리 적은 것. 시간은 경합 상태 값 -> 비율만.
#   AM_TAG (기본 am_nt / am_timed), AM_VARS (기본 전부)
# 출력 /tmp/dkta/lbours/mpm_speed/$TAG/<판> -> 판마다 바로 R2 r2:storage/result/anchorflow/lbours/mpm_speed/$TAG/<판> (x.pt 덤프 포함 통째로). 끝 표식 AM_DONE.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; AF=$(pwd); C=$(git rev-parse --short HEAD)
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
TM=${AM_TIMED:-}; TAG=${AM_TAG:-$([ -n "$TM" ] && echo am_timed || echo am_nt)}
O=/tmp/dkta/lbours/mpm_speed/$TAG; R=r2:storage/result/anchorflow/lbours/mpm_speed/$TAG; mkdir -p $O; rm -f $O/AM_DONE
echo "[커밋] $(git log -1 --oneline) $(date '+%F %T') 태그 $TAG 단독 ${TM:-0}" | tee $O/commit.txt
df -h /tmp | tail -1
declare -A V=(
  [pg]="--mode pg"
  [c0_port]="--mode fast"
  [c1_sort]="--mode fast --sort"
  [c2_fuse]="--mode fast --sort --fuse"
  [c3_sparse]="--mode fast --sort --fuse --sparse"
  [c4_wred]="--mode fast --sort --fuse --sparse --wred"
  [c5_graph]="--mode fast --sort --fuse --sparse --wred --graph"
  [c5_graph_r2]="--mode fast --sort --fuse --sparse --wred --graph"
  [c6_nosort]="--mode fast --fuse --sparse --wred --graph"
  [floor]="--mode floor --graph --no_dump"
)
ORDER=${AM_VARS:-"c5_graph pg c0_port c1_sort c2_fuse c3_sparse c4_wred c5_graph_r2 c6_nosort floor"}
idle() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$1>=1 && $2+0>10{b=1} END{exit b}'; }
G=0
if [ -n "$TM" ]; then
  exec 9>$W/gpu0.lock
else
  exec 9>$W/gpu0.lock
  if flock -w 1800 9; then G=0; echo "[잠금] gpu0.lock $(date +%T) (비측정)"
  else exec 9>&-; G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1!=0' | sort -t, -k2 -n | head -1 | cut -d, -f1)
    echo "[잠금 실패] 카드 $G 에서 비측정 $(date +%T)"; fi
fi
for v in $ORDER; do
  o=$O/$v; [ -f $o/res.json ] && { echo "[건너뜀] $v"; continue; }; mkdir -p $o
  if [ -n "$TM" ]; then
    w=0; until idle; do sleep 20; w=$((w + 20)); done
    flock 9; echo "[잠금] gpu0.lock $(date +%T) 대기 $w s" > $o/TIMED
    until [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -lt 500 ]; do sleep 10; done
  else echo "NOT_TIMED (경합 상태 값, 비율만)" > $o/NOT_TIMED; fi
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $o/gpu_at_start.txt
  echo "커밋 $C 카드 $G ${V[$v]} ${AM_EXTRA:-}" > $o/cmd.txt
  export WARP_CACHE_PATH=/tmp/dkta/warp_cache_am$G CUPY_CACHE_DIR=/tmp/dkta/cupy_cache_am$G; mkdir -p $WARP_CACHE_PATH $CUPY_CACHE_DIR
  (cd $W/i-physgaussian && CUDA_VISIBLE_DEVICES=$G timeout 3600 python -u $AF/exe/mpm_fast.py --out $o ${V[$v]} ${AM_EXTRA:-} < /dev/null > $o/log.txt 2>&1)
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $o/gpu_at_end.txt
  [ -n "$TM" ] && flock -u 9
  echo "$(date +%T) $v $(grep -a '절 AM 결과' $o/log.txt | cut -c1-260) $(grep -a -m1 -i 'error\|Traceback' $o/log.txt)"
  rclone copy $o $R/$v -q
done
flock -u 9 2>/dev/null; exec 9>&-
python -u exe/mpm_fast_cmp.py $O > $O/cmp.txt 2>&1; cat $O/cmp.txt
echo "AM_DONE $(date +%T)"; touch $O/AM_DONE; rclone copy $O $R -q
