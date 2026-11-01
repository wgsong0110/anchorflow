#!/bin/bash
# 절 AL (wolf 재질 판 속도): wolf AG.19 공통 설정 한 회를 GPU0 에서 -- gpu0.lock 을 쥔 채 (다른 갈래 단독 측정과 직렬화), GPU0 이 빌 때까지 기다린 뒤.
#   카드 1~6 이 차 있으면 비측정 (fem_ab_run.sh 의 NOT_TIMED, 구간 비율만 읽는다). 11-02 10:00 KST 뒤에는 새로 시작하지 않는다.
# 사용: bash exe/fem_al_run.sh <태그> [추가 인자...]   (출력 /tmp/dkta/lbours/fem_al_wolf/<태그>, R2 같은 경로)
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; TG=${1:?태그}; shift
[ $(date +%s) -ge $(TZ=Asia/Seoul date -d "2026-11-02 10:00" +%s) ] && { echo "[중단] 10:00 KST 뒤 새 실행 금지"; exit 1; }
BASE="--data /tmp/dkta/ag_data/wolf.pt --tag wolf --frames 50 --node_h 0.014142 --floor vel --pbc --plast --mode lag --opt --mass tet --rstat --lim_sr 5 --cfuse --cfuse_mode 2 --sp vcap --sp_vcap 0.5 --no_dump --N 640 --lim_sr 1e30 --sp_vcap 0.5 --ak_pq --ak_capmu --akpl_el --ak_sfcr --ah_mcfl 1 --ak_pq_plast inc --vfl_w gtk --vfl_rate --vfl_npd 0.005 --vfl_band 0 --node_h 0.028284"
df -h /tmp | tail -1
exec 9>$W/gpu0.lock
flock 9
echo "[잠금] $(date '+%F %T') $TG"
while [ "$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 500 ]; do sleep 10; done
# fem_ab_run.sh 와 같은 기록 (cmd.txt·rc.txt·gpu_at_*·err_fem.txt·EX_DONE, R2 2 분 동기화) -- 단 conda activate 대신 env 경로를 바로, CuPy 커널 캐시·바이트코드는 노드 로컬
#   (11-01 NFS /home 100 % 로 conda activate·첫 컴파일 캐시 쓰기가 수십 분 D 상태로 멈췄다)
export PYTHONUTF8=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 CUPY_CACHE_DIR=/tmp/dkta/cupy_cache_al
export PATH=/home/dkta/.conda/envs/af/bin:$HOME/bin:$PATH
C=$(git rev-parse --short HEAD); O=/tmp/dkta/lbours/fem_al_wolf/$TG; R=r2:storage/result/anchorflow/lbours/fem_al_wolf/$TG; mkdir -p $O
( while true; do sleep 120; rclone copy $O $R -q; done ) & SYNC=$!
echo "[비측정] 카드 1~6 상태는 gpu_at_start.txt -- ms 는 구간 비율로만" > $O/NOT_TIMED
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $O/gpu_at_start.txt
echo "[명령] 커밋 $C CUDA_VISIBLE_DEVICES=0 python -u exe/fem_ex_sim.py --out $O $BASE $*" | tee $O/cmd.txt
CUDA_VISIBLE_DEVICES=0 python -u exe/fem_ex_sim.py --out $O $BASE "$@" < /dev/null > $O/v_fem.log 2>&1
echo "rc=$? $(date '+%F %T')" > $O/rc.txt
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > $O/gpu_at_end.txt
grep -a "롤아웃\]\|\[CFL\]\|\[발산\]" $O/v_fem.log | tail -3 > $O/err_fem.txt
if [ -f $O/summary.json ]; then touch $O/EX_DONE; else grep -a "Traceback\|Error\|error" $O/v_fem.log | tail -5 > $O/EX_FAIL; fi
kill $SYNC 2>/dev/null; rclone copy $O $R -q
flock -u 9; exec 9>&-
