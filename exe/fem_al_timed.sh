#!/bin/bash
# 절 AL 단독 측정 (다음 잡용 -- 잡 129982 에서는 띄우지 않는다): 기준 판과 --al_fuse --al_noxp 판을 번갈아 3 회씩, 회마다 gpu0.lock + 카드 1~6 이 모두 빈 뒤 GPU0.
#   태그 al_t{b,x}<회>, 출력 /tmp/dkta/lbours/fem_al_wolf/ (R2 같은 경로). 끝 표식 al_timed_DONE (R2 에도).
# 사용 (잡 안에서, exe/job_run.sh AF_DETACH=1 로): WL_C=<커밋> bash /home/dkta/work/wolf_boot.sh fem_al_timed.sh
cd "$(dirname "$0")/.." || exit 1
export AL_TIMED=1 AL_NOT_AFTER=""
echo "[단독 AL] 커밋 $(git log -1 --oneline) $(date '+%F %T')"
for r in 1 2 3; do
  bash exe/fem_al_run.sh al_tb$r --secprof
  bash exe/fem_al_run.sh al_tx$r --secprof --al_fuse --al_noxp
  for t in al_tb$r al_tx$r; do O=/tmp/dkta/lbours/fem_al_wolf/$t; echo "$(date +%T) $t $(cat $O/err_fem.txt | tail -1 | cut -c1-160) | 시작 $(tr '\n' ' ' < $O/gpu_at_start.txt)"; done
done
O=/tmp/dkta/lbours/fem_al_wolf; touch $O/al_timed_DONE; /home/dkta/bin/rclone copy $O/al_timed_DONE r2:storage/result/anchorflow/lbours/fem_al_wolf -q
