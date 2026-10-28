#!/bin/bash
# 명시적 FEM 실행 (doc/fem_realtime.md 절 L, exe/fem_ex_sim.py). 결과는 lbours/${EX_ROOT:-fem_ex_lego}/<태그> 에 쓰고 R2 로 동기화 (실행 중 2 분 간격 + 끝).
# 사용: bash exe/fem_ex_run.sh <태그> <GPU> [fem_ex_sim.py 인자...]      표식 EX_DONE / EX_FAIL.  이미 EX_DONE 이면 건너뛴다.
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
W=/home/dkta/work; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
TG=${1:?태그}; G=${2:?GPU}; shift 2
cd "$(dirname "$0")/.." || exit 1
C=$(git rev-parse --short HEAD)
EX_ROOT=${EX_ROOT:-fem_ex_lego}
O=$W/lbours/$EX_ROOT/$TG; R=r2:storage/result/anchorflow/lbours/$EX_ROOT/$TG; mkdir -p $O
[ -f $O/EX_DONE ] && { echo "[건너뜀] $TG"; cat $O/err_fem.txt; exit 0; }
rm -f $O/EX_FAIL
( while true; do sleep 120; rclone copy $O $R -q; done ) & SYNC=$!
trap 'kill $SYNC 2>/dev/null' EXIT
# 속도 측정은 단독: 노드 7 장 모두 빌 때까지 기다린다 (EX_NOWAIT=1: 오차만 보는 비측정 실행 -- 다른 카드와 병렬, cmd.txt 에 표시)
if [ -z "${EX_NOWAIT:-}" ]; then
while [ -n "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 500')" ]; do echo "[대기] GPU 사용 중 $(date +%T)"; sleep 20; done
else echo "[비측정] 병렬 실행 -- ms 는 쓰지 않는다" > $O/NOT_TIMED; fi
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $O/gpu_at_start.txt
echo "[명령] 커밋 $C CUDA_VISIBLE_DEVICES=$G python -u exe/fem_ex_sim.py --out $O $*" | tee $O/cmd.txt
CUDA_VISIBLE_DEVICES=$G python -u exe/fem_ex_sim.py --out $O "$@" < /dev/null > $O/v_fem.log 2>&1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $O/gpu_at_end.txt
grep -a "롤아웃\]\|\[CFL\]\|\[발산\]" $O/v_fem.log | tail -3 > $O/err_fem.txt
if [ -f $O/summary.json ]; then touch $O/EX_DONE; else grep -a "Traceback\|Error\|error" $O/v_fem.log | tail -5 > $O/EX_FAIL; cat $O/EX_FAIL; fi
cat $O/err_fem.txt
kill $SYNC 2>/dev/null
rclone copy $O $R -q
