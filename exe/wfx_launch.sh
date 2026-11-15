#!/bin/bash
# wolffix (11-15): vast 인스턴스 하나 = exe/vast_wfx.sh. 사용: bash exe/wfx_launch.sh <offer> <태그> <커밋> <R2 명세 디렉토리> [디스크GB]
#  R2 키는 로컬 rclone 설정에서 읽어 --env 로만 넘김 (파일에 안 씀). 라벨 wolffix_<태그>.
set -e
OFFER=$1; TAG=$2; H=$3; SPEC=$4; DISK=${5:-120}
AK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['access_key_id'])")
SK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['secret_access_key'])")
EP=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['endpoint'])")
CODE=r2:storage/result/anchorflow/lbours/wolffix1115/code
ONS="bash -c '(command -v unzip >/dev/null && command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux curl git unzip) >/dev/null 2>&1); command -v rclone >/dev/null || (curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/); mkdir -p /root/.config/rclone; printf \"[r2]\\ntype = s3\\nprovider = Cloudflare\\naccess_key_id = %s\\nsecret_access_key = %s\\nendpoint = %s\\nacl = private\\nno_check_bucket = true\\n\" \"\$R2_ACCESS_KEY\" \"\$R2_SECRET\" \"\$R2_ENDPOINT\" > /root/.config/rclone/rclone.conf; rclone copyto $CODE/vast_wfx.sh /root/vast_wfx.sh; env | grep -E \"^(R2_|XS_|CONTAINER_)\" | sed \"s/^/export /;s/=/=\\\"/;s/\$/\\\"/\" > /root/xsenv.sh; tmux has-session -t sv 2>/dev/null || tmux new-session -d -s sv \"source /root/xsenv.sh; bash /root/vast_wfx.sh > /root/vast_wfx.out 2>&1\"'"
vastai create instance $OFFER --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel --disk $DISK --ssh --direct --label "wolffix_$TAG" \
  --env "-e R2_ACCESS_KEY=$AK -e R2_SECRET=$SK -e R2_ENDPOINT=$EP -e XS_TAG=$TAG -e XS_COMMIT=$H -e XS_SPEC=$SPEC" --onstart-cmd "$ONS" --raw
