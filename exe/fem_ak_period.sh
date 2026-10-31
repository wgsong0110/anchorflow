#!/bin/bash
# 절 AK.16: ficus 잎 끝 진동 주기 (exe/fem_p_period.py, 절 AD 와 같은 식: z 상위 5 % 입자, 정답 주 방향, 영점 교차 8 개 x 2 / FFT). 사용: bash exe/fem_ak_period.sh <R2 lbours 상대 경로 (덤프 폴더)> ...
cd "$(dirname "$0")/.." || exit 1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
PS=()
for rel in "$@"; do
  O=/tmp/dkta/lbours/$rel
  if [ ! -f $O/roll_fem.pt ] && [ ! -f $O/pred.pt ]; then mkdir -p $O; rclone copy r2:storage/result/anchorflow/lbours/$rel/roll_fem.pt $O -q; fi
  [ -f $O/pred.pt ] && PS+=($O/pred.pt) || PS+=($O/roll_fem.pt)
done
python -u exe/fem_p_period.py --data /home/dkta/work/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --pred "${PS[@]}"
for rel in "$@"; do rclone copy /tmp/dkta/lbours/$rel/period.json r2:storage/result/anchorflow/lbours/$rel -q; done
echo "[끝] $(date +%T)"
