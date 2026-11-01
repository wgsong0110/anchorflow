#!/bin/bash
# 절 AK 깨끗한 단독 측정: 작업 목록 (fem_ak_jobs.py, 보통 v:<판>:<바닥>:<회>::_t 꼴 + AK_KIND=t) 을 차례로 GPU0 에서 -- 측정 카드 (GPU0) 만 빌 때 (fem_ab_run.sh 의 gpu0.lock + GPU0 ≤ 500 MiB 대기; 11-01 정정: 카드 1 에서 6 조건 없앰) 
# fem_ab_run.sh EX_TIMED=1 (gpu0.lock flock, GPU0 빌 때까지). 시작·끝 7 장 상태 기록. 표식 /tmp/dkta/lbours/fem_ak_logs/TIMED_DONE_<이름>.
# 사용: bash exe/fem_ak_timed.sh <단계> <씬,...> <이름>
#   AK_NOT_BEFORE="2026-11-02 10:40" (KST) 이면 그 시각 전에는 시작하지 않는다 (측정 창); AK_SKIP_DONE=1 이면 이미 단독 측정이 끝난 출력 (gpu_at_end.txt 있음) 은 건너뛴다.
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
ST=${1:?단계}; SCS=${2:?씬}; NM=${3:?이름}
OD=/tmp/dkta/lbours/fem_ak_logs; mkdir -p $OD
echo "[단독 $ST] 커밋 $(git log -1 --oneline) $(date '+%F %T')" | tee -a $OD/timed_$NM.log
while read -r ROOT TG ARGS; do
  O=/tmp/dkta/lbours/$ROOT/$TG
  [ "${AK_SKIP_DONE:-0}" = 1 ] && [ -f $O/gpu_at_end.txt ] && grep -qa '롤아웃\]' $O/v_fem.log 2>/dev/null && continue
  if [ -n "${AK_NOT_BEFORE:-}" ]; then G=$(TZ=Asia/Seoul date -d "$AK_NOT_BEFORE" +%s); while [ $(date +%s) -lt $G ]; do sleep 60; done; fi
  EX_TIMED=1 EX_ROOT=$ROOT bash exe/fem_ab_run.sh "$TG" 0 $ARGS < /dev/null > /dev/null 2>&1
  echo "$(date +%T) $TG $(grep -a '롤아웃\]\|\[발산\]' $O/v_fem.log | tail -1 | cut -c1-170) | 시작 $(tr '\n' ' ' < $O/gpu_at_start.txt) | 끝 $(tr '\n' ' ' < $O/gpu_at_end.txt)" | tee -a $OD/timed_$NM.log
done < <(AK_KIND=t python exe/fem_ak_jobs.py "$ST" "$SCS")
echo "[단독 끝] $(date +%T)" | tee -a $OD/timed_$NM.log; touch $OD/TIMED_DONE_$NM
rclone copy $OD r2:storage/result/anchorflow/lbours/fem_ak_logs -q
