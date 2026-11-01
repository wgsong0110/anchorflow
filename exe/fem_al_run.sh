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
EX_ROOT=fem_al_wolf bash exe/fem_ab_run.sh "$TG" 0 $BASE "$@" < /dev/null
flock -u 9; exec 9>&-
