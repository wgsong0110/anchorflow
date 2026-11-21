#!/bin/bash
# 맨 뒤 반복 측정 (파레토 2·3 회차, 2^3 정당화 r2·r3·시간 2·3 회): vast 인스턴스 하나 띄우기 (로컬에서 실행).
#  사용: bash exe/rep_launch.sh <offer> <라벨> <R2 부팅 스크립트 경로> "<추가 env: -e K=V ...>"
#  onstart = rclone 설치·설정 -> R2 의 부팅 스크립트를 받아 tmux main 에서 실행. R2 키는 로컬 rclone 설정에서 읽어 --env 로만 (파일에 안 씀).
set -e
OFFER=$1; LBL=$2; BOOT=$3; XENV=${4:-}
AK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['access_key_id'])")
SK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['secret_access_key'])")
EP=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['endpoint'])")
ONS="bash -c '(command -v unzip >/dev/null && command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux curl git unzip) >/dev/null 2>&1); command -v rclone >/dev/null || (curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/); mkdir -p /root/.config/rclone; printf \"[r2]\\ntype = s3\\nprovider = Cloudflare\\naccess_key_id = %s\\nsecret_access_key = %s\\nendpoint = %s\\nacl = private\\nno_check_bucket = true\\n\" \"\$R2_ACCESS_KEY\" \"\$R2_SECRET\" \"\$R2_ENDPOINT\" > /root/.config/rclone/rclone.conf; rclone copyto $BOOT /root/rep_boot.sh; env | grep -E \"^(R2_|GQ_|MB_|SGR_|REP_|CONTAINER_)\" | sed \"s/^/export /;s/=/=\\\"/;s/\$/\\\"/\" > /root/repenv.sh; tmux has-session -t main 2>/dev/null || tmux new-session -d -s main \"source /root/repenv.sh; bash /root/rep_boot.sh > /root/rep_boot.out 2>&1\"'"
vastai create instance $OFFER --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel --disk ${DISK:-150} --ssh --direct --label "$LBL" \
  --env "-e R2_ACCESS_KEY=$AK -e R2_SECRET=$SK -e R2_ENDPOINT=$EP -e REP_TAG=$LBL $XENV" --onstart-cmd "$ONS" --raw
