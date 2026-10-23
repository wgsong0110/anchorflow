#!/bin/bash
# A5 드라이버 기동 (계산 노드 잡 안에서만, exe/job_run.sh 로). 잡이 끝나 죽었으면 새 잡에서 같은 명령 한 줄로 다시 띄운다:
#   AF_DETACH=1 bash /home/dkta/work/anchorflow/exe/job_run.sh /home/dkta/work/stab_a5_go.sh /home/dkta/work/stab/a5/driver.out
# 코드 판은 /home/dkta/work/stab/a5/COMMIT 에 고정 (처음 띄울 때 적어 둔 것). 이미 돌고 있으면 그냥 나간다.
W=/home/dkta/work; A=$W/stab/a5; mkdir -p $A
C=$(cat $A/COMMIT 2>/dev/null); [ -n "$C" ] || { echo "COMMIT 파일 없음"; exit 1; }
exec 9>$A/driver.lock; flock -n 9 || { echo "[중단] 드라이버가 이미 돌고 있다"; exit 0; }
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af
WT=$W/wt/stab_a5_$C
[ -d $WT ] || (cd $W/anchorflow && git fetch -q origin && git worktree add -q --detach $WT $C)
[ -f $WT/exe/stab_a5_driver.py ] || { echo "워크트리 이상 $WT"; exit 1; }
python -c "import lpips, skimage" 2>/dev/null || pip install -q lpips scikit-image
echo "[기동] $(date '+%F %T') $(hostname) 잡 ${PBS_JOBID:-?} 커밋 $C"
exec python -u $WT/exe/stab_a5_driver.py
