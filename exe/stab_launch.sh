#!/bin/bash
# 안정 서브스텝 묶음 로컬 실행기 (exe/pareto_launch.sh 복사, doc/pareto.md 절 S): 지금 커밋을 git bundle -> R2 lbours/stab/code -> 진입점 (stab_go_tmpl.sh) 을 /home/dkta/work/stab/ 로 -> job_run.sh.
# 사용: bash exe/stab_launch.sh <이름> '<잡 안 명령>'   (PA_DETACH=1 이면 떼어 띄움; PA_JOB·PA_NODE)
set -e
cd "$(dirname "$0")/.."
NM=$1; CMD=$2; C=$(git rev-parse --short HEAD)
J=${PA_JOB:?잡 id (예 130784.ECE-util1)}; N=${PA_NODE:?노드}
T=/tmp/stab_code; mkdir -p $T
[ -f $T/imp_$C.bundle ] || git bundle create $T/imp_$C.bundle HEAD implicit-bspline -q 2>/dev/null || git bundle create $T/imp_$C.bundle HEAD implicit-bspline
rclone copyto $T/imp_$C.bundle r2:storage/result/anchorflow/lbours/stab/code/imp_$C.bundle -q
CE=$(printf '%s' "$CMD" | sed -e 's/[\/&|]/\\&/g')
sed -e "s|@C@|$C|g" -e "s|@NAME@|$NM|g" -e "s|@CMD@|$CE|g" exe/stab_go_tmpl.sh > $T/go_$NM.sh
H=dkta@147.46.121.39
ssh $H "mkdir -p /home/dkta/work/stab" || ssh $H "mkdir -p /home/dkta/work/stab"
scp -q $T/go_$NM.sh $H:/home/dkta/work/stab/go_$NM.sh || scp -q $T/go_$NM.sh $H:/home/dkta/work/stab/go_$NM.sh
if [ -n "${PA_DETACH:-}" ]; then DT="AF_DETACH=1"; else DT=""; fi
ssh $H "$DT AF_JOB=$J AF_NODE=$N bash /home/dkta/work/anchorflow/exe/job_run.sh /home/dkta/work/stab/go_$NM.sh /home/dkta/work/stab/go_$NM.out" \
  || ssh $H "$DT AF_JOB=$J AF_NODE=$N bash /home/dkta/work/anchorflow/exe/job_run.sh /home/dkta/work/stab/go_$NM.sh /home/dkta/work/stab/go_$NM.out"
echo "[띄움] $NM 커밋 $C"
