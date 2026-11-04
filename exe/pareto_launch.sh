#!/bin/bash
# 파레토 묶음 로컬 실행기: 지금 커밋을 git bundle -> R2 lbours/pareto/code -> 진입점 (pareto_go_tmpl.sh) 을 /home/dkta/work/pareto/ 로 -> job_run.sh.
# 사용: bash exe/pareto_launch.sh <이름> '<잡 안 명령>'   (PA_DETACH=1 이면 떼어 띄움; PA_JOB·PA_NODE)
set -e
cd "$(dirname "$0")/.."
NM=$1; CMD=$2; C=$(git rev-parse --short HEAD)
J=${PA_JOB:-130567.ECE-util1}; N=${PA_NODE:-ece-a6gpu10}
T=/tmp/pareto_code; mkdir -p $T
[ -f $T/imp_$C.bundle ] || git bundle create $T/imp_$C.bundle HEAD implicit-bspline -q 2>/dev/null || git bundle create $T/imp_$C.bundle HEAD implicit-bspline
rclone copyto $T/imp_$C.bundle r2:storage/result/anchorflow/lbours/pareto/code/imp_$C.bundle -q
CE=$(printf '%s' "$CMD" | sed -e 's/[\/&|]/\\&/g')
sed -e "s|@C@|$C|g" -e "s|@NAME@|$NM|g" -e "s|@CMD@|$CE|g" exe/pareto_go_tmpl.sh > $T/go_$NM.sh
H=dkta@147.46.121.39
ssh $H "mkdir -p /home/dkta/work/pareto" || ssh $H "mkdir -p /home/dkta/work/pareto"
scp -q $T/go_$NM.sh $H:/home/dkta/work/pareto/go_$NM.sh || scp -q $T/go_$NM.sh $H:/home/dkta/work/pareto/go_$NM.sh
if [ -n "${PA_DETACH:-}" ]; then DT="AF_DETACH=1"; else DT=""; fi
ssh $H "$DT AF_JOB=$J AF_NODE=$N bash /home/dkta/work/anchorflow/exe/job_run.sh /home/dkta/work/pareto/go_$NM.sh /home/dkta/work/pareto/go_$NM.out" \
  || ssh $H "$DT AF_JOB=$J AF_NODE=$N bash /home/dkta/work/anchorflow/exe/job_run.sh /home/dkta/work/pareto/go_$NM.sh /home/dkta/work/pareto/go_$NM.out"
echo "[띄움] $NM 커밋 $C"
