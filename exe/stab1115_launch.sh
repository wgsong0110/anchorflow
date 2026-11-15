#!/bin/bash
# [stab1115] vast 인스턴스 하나 띄우기. 사용: bash exe/stab1115_launch.sh <offer id> <태그> '<씬:단계+씬:단계>' [커밋] [동시 판 수]
#  R2 키는 로컬 rclone 설정에서 읽어 --env 로만 넘김 (파일에 안 씀). 부팅 스크립트 = R2 lbours/stab1115/code/boot_<커밋>.sh
set -e
OFFER=$1; TAG=$2; JOBS=$3; C=${4:-$(cat /tmp/s1115/commit.txt)}; PAR=${5:-4}
AK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['access_key_id'])")
SK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['secret_access_key'])")
EP=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['endpoint'])")
ONS="bash -c '(command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux curl git unzip) >/dev/null 2>&1); command -v rclone >/dev/null || (curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/); mkdir -p /root/.config/rclone; printf \"[r2]\\ntype = s3\\nprovider = Cloudflare\\naccess_key_id = %s\\nsecret_access_key = %s\\nendpoint = %s\\nacl = private\\nno_check_bucket = true\\n\" \"\$R2_ACCESS_KEY\" \"\$R2_SECRET\" \"\$R2_ENDPOINT\" > /root/.config/rclone/rclone.conf; rclone copyto r2:storage/result/anchorflow/lbours/stab1115/code/boot_$C.sh /root/boot.sh; env | grep -E \"^(R2_|S1115_|CONTAINER_)\" | sed \"s/^/export /\" > /root/s1env.sh; tmux has-session -t s1 2>/dev/null || tmux new-session -d -s s1 \"source /root/s1env.sh; bash /root/boot.sh > /root/boot.out 2>&1\"'"
vastai create instance $OFFER --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel --disk 200 --ssh --direct --label "s1115_$TAG" \
  --env "-e R2_ACCESS_KEY=$AK -e R2_SECRET=$SK -e R2_ENDPOINT=$EP -e S1115_COMMIT=$C -e S1115_JOBS=$JOBS -e S1115_TAG=$TAG -e S1115_PAR=$PAR" --onstart-cmd "$ONS" --raw
