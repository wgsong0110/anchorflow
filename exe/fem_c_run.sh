#!/bin/bash
# 절 C 실행 (exe/fem_ex_sim.py). fem_ex_run.sh 와 같고 측정 방식만 다르다:
#   EX_TIMED=1 : 단독 측정 -- GPU0 만, /home/dkta/work/gpu0.lock 을 flock 으로 잡고 (다른 갈래 측정과 직렬화) GPU0 이 빌 때까지 기다린 뒤 실행.
#                시작·끝 7 장 상태 기록 (다른 카드의 비측정 실행은 있을 수 있다 -- gpu_at_start.txt).
#   그 밖     : 비측정 (NOT_TIMED 표식, ms 는 쓰지 않는다).
# 결과 lbours/${EX_ROOT}/<태그>, R2 같은 경로로 2 분 간격 + 끝 동기화. 사용: bash exe/fem_c_run.sh <태그> <GPU> [fem_ex_sim.py 인자...]
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
TG=${1:?태그}; G=${2:?GPU}; shift 2
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
EX_ROOT=${EX_ROOT:-fem_c_lego}
O=$W/lbours/$EX_ROOT/$TG; R=r2:storage/result/anchorflow/lbours/$EX_ROOT/$TG; mkdir -p $O
[ -f $O/EX_DONE ] && { echo "[건너뜀] $TG"; cat $O/err_fem.txt; exit 0; }
rm -f $O/EX_FAIL
( while true; do sleep 120; rclone copy $O $R -q; done ) & SYNC=$!
trap 'kill $SYNC 2>/dev/null' EXIT
run() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $O/gpu_at_start.txt
  echo "[명령] 커밋 $C CUDA_VISIBLE_DEVICES=$G python -u exe/fem_ex_sim.py --out $O $*" | tee $O/cmd.txt
  CUDA_VISIBLE_DEVICES=$G python -u exe/fem_ex_sim.py --out $O "$@" < /dev/null > $O/v_fem.log 2>&1
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $O/gpu_at_end.txt
}
if [ -n "${EX_TIMED:-}" ]; then
  [ "$G" = 0 ] || { echo "[오류] 단독 측정은 GPU0 만"; exit 1; }
  exec 9>$W/gpu0.lock
  flock 9
  echo "[잠금] gpu0.lock $(date +%T)" > $O/TIMED
  while [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 500 ]; do sleep 10; done
  run "$@"
  flock -u 9; exec 9>&-
else
  echo "[비측정] 병렬 실행 -- ms 는 쓰지 않는다" > $O/NOT_TIMED
  run "$@"
fi
grep -a "롤아웃\]\|\[CFL\]\|\[발산\]" $O/v_fem.log | tail -3 > $O/err_fem.txt
if [ -f $O/summary.json ]; then touch $O/EX_DONE; else grep -a "Traceback\|Error\|error" $O/v_fem.log | tail -5 > $O/EX_FAIL; cat $O/EX_FAIL; fi
cat $O/err_fem.txt
kill $SYNC 2>/dev/null
rclone copy $O $R -q
