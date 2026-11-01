#!/bin/bash
# 절 AM: 정답 MPM (lego L000, PG warp 솔버 정답 설정 그대로) 가속 판들을 한 프로세스에서 차례로 (exe/mpm_fast_multi.py) 돌리고 결과 불변을 비교한다.
#   AM_TIMED=1 AM_GPU=<카드> : 단독 측정 -- 판마다 gpu<카드>.lock 을 잡고 그 카드에 다른 계산 프로세스가 없을 때까지 기다린 뒤 실행 (11-02 정정: 다른 카드는 상관없다),
#     실행 중 5 초마다 그 카드 계산 프로세스 표본 -> TIMED 또는 TIMED_CONTAMINATED 표식
#   기본 (비측정, NOT_TIMED): gpu0.lock 을 최대 30 분 기다려 GPU0 (다른 갈래 단독 측정과 직렬화), 못 잡으면 카드 1~6 중 메모리 적은 것. 시간은 경합 상태 값 -> 비율만.
#   AM_TAG (기본 am_nt / am_timed), AM_VARS (기본 전부), AM_EXTRA (mpm_fast.py 에 덧붙일 인자)
# 출력 /tmp/dkta/lbours/mpm_speed/$TAG/<판> -> 판마다 바로 R2 r2:storage/result/anchorflow/lbours/mpm_speed/$TAG/<판> (x.pt 덤프 포함 통째로). 끝 표식 AM_DONE.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; AF=$(pwd)
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
TM=${AM_TIMED:-}; TAG=${AM_TAG:-$([ -n "$TM" ] && echo am_timed || echo am_nt)}
O=/tmp/dkta/lbours/mpm_speed/$TAG; R=r2:storage/result/anchorflow/lbours/mpm_speed/$TAG; mkdir -p $O; rm -f $O/AM_DONE
echo "[커밋] $(git log -1 --oneline) $(date '+%F %T') 태그 $TAG 단독 ${TM:-0}" | tee $O/commit.txt
df -h /tmp | tail -1
VARS=${AM_VARS:-"c8_shared pg c0_port c1_sort c2_fuse c3_sparse c4_wred c5_graph c5_b256 c6_nosort c7_nowred c8_shared_nowred c8_shared_r2 floor"}
export WARP_CACHE_PATH=/tmp/dkta/warp_cache_am CUPY_CACHE_DIR=/tmp/dkta/cupy_cache_am; mkdir -p $WARP_CACHE_PATH $CUPY_CACHE_DIR
if [ -n "$TM" ]; then
  G=${AM_GPU:-0}; TF="--timed --gpu $G"
else
  TF=; exec 9>$W/gpu0.lock
  if flock -w 1800 9; then G=0; echo "[잠금] gpu0.lock $(date +%T) (비측정)"
  else exec 9>&-; G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1!=0' | sort -t, -k2 -n | head -1 | cut -d, -f1)
    echo "[잠금 실패] 카드 $G 에서 비측정 $(date +%T)"; fi
fi
(cd $W/i-physgaussian && CUDA_VISIBLE_DEVICES=$G python -u $AF/exe/mpm_fast_multi.py --out $O --vars "$VARS" --extra "${AM_EXTRA:-}" $TF --r2 $R < /dev/null 2>&1) | tee $O/multi.log
flock -u 9 2>/dev/null; exec 9>&- 2>/dev/null
CUDA_VISIBLE_DEVICES= python -u exe/mpm_fast_cmp.py $O > $O/cmp.txt 2>&1; cat $O/cmp.txt
echo "AM_DONE $(date +%T)"; touch $O/AM_DONE; rclone copy $O $R -q
