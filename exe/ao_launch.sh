#!/bin/bash
# 절 AO 로컬 실행기: 커밋 C 의 git archive 를 R2 로 올리고, 진입점 (ao_go_tmpl.sh 치환) 을 클러스터 NFS (/home/dkta/work/ao/) 로 보내 job_run.sh (AF_DETACH=1) 로 띄운다.
# 사용: bash exe/ao_launch.sh <이름> '<잡 안 명령 (풀린 커밋 디렉토리에서)>'   AO_JOB·AO_NODE 로 잡 지정 (기본 129982 / ece-a6gpu10)
set -e
cd "$(dirname "$0")/.."
NM=$1; CMD=$2; C=$(git rev-parse --short HEAD)
J=${AO_JOB:-129982.ECE-util1}; N=${AO_NODE:-ece-a6gpu10}
T=/tmp/ao_code; mkdir -p $T
[ -f $T/ao_$C.tar.gz ] || git archive --format=tar.gz -o $T/ao_$C.tar.gz HEAD
rclone copyto $T/ao_$C.tar.gz r2:storage/result/anchorflow/lbours/fem_ao/code/ao_$C.tar.gz -q
CE=$(printf '%s' "$CMD" | sed -e 's/[\/&|]/\\&/g')
sed -e "s|@C@|$C|g" -e "s|@NAME@|$NM|g" -e "s|@CMD@|$CE|g" exe/ao_go_tmpl.sh > $T/go_$NM.sh
H=dkta@147.46.121.39
ssh $H "mkdir -p /home/dkta/work/ao" || ssh $H "mkdir -p /home/dkta/work/ao"
for k in 1 2; do scp -q $T/go_$NM.sh $H:/home/dkta/work/ao/go_$NM.sh && break; sleep 5; done
for k in 1 2; do ssh $H "AF_DETACH=1 AF_JOB=$J AF_NODE=$N bash /home/dkta/work/anchorflow/exe/job_run.sh /home/dkta/work/ao/go_$NM.sh /home/dkta/work/ao/go_$NM.out" && break; sleep 5; done
echo "[띄움] $NM 커밋 $C"
