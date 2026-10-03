#!/bin/bash
# 클러스터 잡 안에서 스크립트를 실행한다. **대화형 판에 타이핑하지 않는다.**
#
# 왜 이것만 쓰는가 (2026-09-30, 10-01 사고 두 번):
#   대화형 판(`qsub -I` 세션)에 명령을 타이핑하면 두 가지로 망가진다.
#   (1) 실행 중인 판에 보내면 텍스트가 돌고 있는 프로세스의 stdin 으로 삼켜져
#       조용히 사라진다 (실행된 줄 알고 몇 시간을 날린다).
#   (2) taichi/warp 를 쓰는 프로세스가 터미널을 물고 죽으면 판이 입력을 영구히
#       받지 않게 된다. 원격 tty 는 멀쩡한데 qsub 클라이언트의 입력 전달이
#       끊겨 원격에서 고칠 수 없다. 9-30 에 이것으로 A6000 7 장 잡을 날렸다.
#
# 해결: PBS 공식 도구 `pbs_attach` 로 **잡 문맥에 붙여** 실행한다. 잡의 세션에
# 들어가므로 계산이 그 잡으로 계리되고, torch 가 CUDA 를 정상으로 본다
# (실측: torch 2.5.1+cu121, cuda True, 7 장). 이 잡은 `ssu_whole_a6gpu -g=7` 로
# **노드 통째**라 남의 카드를 쓸 위험이 없는 경우다 (1 장짜리 잡에는 쓰지 말 것).
#
# 사용:
#   bash exe/job_run.sh <실행할_스크립트> [로그파일]
#   AF_JOB=126703.ECE-util1 AF_NODE=ece-a6gpu10 로 대상을 바꾼다
set -u
J=${AF_JOB:-}
N=${AF_NODE:-}
S=${1:?실행할 스크립트를 주세요}
L=${2:-/home/dkta/work/job_run.out}
if [ -z "$J" ] || [ -z "$N" ]; then
  # 돌고 있는 내 잡을 찾는다
  read -r J N <<EOF
$(/opt/pbs/bin/qstat -answ1 -u "$(whoami)" 2>/dev/null \
  | awk '$10=="R"{split($12,a,"/"); print $1, a[1]; exit}')
EOF
fi
[ -z "${J:-}" ] && { echo "돌고 있는 잡이 없다"; exit 1; }
echo "[잡] $J @ $N  ->  $S  (로그 $L)"
# stdin 을 끊는다. 예외 없다 -- 이것이 판을 망가뜨린 원인이었다.
#
# AF_DETACH=1 이면 **원격에서 떼어내** 띄우고 바로 돌아온다. 긴 잡을 ssh 로
# 붙들고 있으면 이쪽 타임아웃에 ssh 가 끊길 때 원격까지 SIGHUP 으로 같이
# 죽는다 (실측: 89 프레임 실행 두 건이 580 초에서 함께 사라졌다).
# 떼어낸 뒤에는 **로그 파일로** 진행을 확인한다.
if [ -n "${AF_DETACH:-}" ]; then
  ssh -o ConnectTimeout=15 "$N" \
    "setsid nohup /opt/pbs/bin/pbs_attach -j $J /bin/bash $S < /dev/null \
       > $L 2>&1 & echo 떼어냄 pid=\$!" < /dev/null
  rc=$?
else
ssh -o ConnectTimeout=15 "$N" \
  "/opt/pbs/bin/pbs_attach -j $J /bin/bash $S < /dev/null > $L 2>&1" \
  < /dev/null
rc=$?
fi
echo "[완료] rc=$rc"
# 실행 여부를 **출력 파일로 확인**한다 (보냈다고 돌았다고 믿지 않는다)
ssh -o ConnectTimeout=15 "$N" "ls -la $L" < /dev/null 2>&1 | tail -1
exit $rc
