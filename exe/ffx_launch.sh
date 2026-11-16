#!/bin/bash
# 절 CM191: vast 인스턴스 하나 (라벨 final_fix_<이름>) = 장면 묶음 하나. 사용: bash exe/fpa_launch.sh <offer> <이름> <장면쉼표> <커밋>
#  대기열은 R2 lbours/final_fix/marks/final_fix_<이름>.queue (미리 올림). R2 키는 로컬 rclone 설정에서 읽어 --env 로만 넘김 (파일에 안 씀).
set -e
OFFER=$1; NAME=$2; SC=$3; H=$4; TAG=final_fix_$NAME; CODE=r2:storage/result/anchorflow/lbours/final_fix/code
AK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['access_key_id'])")
SK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['secret_access_key'])")
EP=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['endpoint'])")
ONS="bash -c '(command -v unzip >/dev/null && command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux curl git unzip) >/dev/null 2>&1); command -v rclone >/dev/null || (curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/); mkdir -p /root/.config/rclone; printf \"[r2]\\ntype = s3\\nprovider = Cloudflare\\naccess_key_id = %s\\nsecret_access_key = %s\\nendpoint = %s\\nacl = private\\nno_check_bucket = true\\n\" \"\$R2_ACCESS_KEY\" \"\$R2_SECRET\" \"\$R2_ENDPOINT\" > /root/.config/rclone/rclone.conf; rclone copyto $CODE/vast_ffx_$H.sh /root/vast_ffx.sh; env | grep -E \"^(R2_|FPA_|FFX_|CONTAINER_)\" | sed \"s/^/export /;s/=/=\\\"/;s/\$/\\\"/\" > /root/xsenv.sh; tmux has-session -t ffx 2>/dev/null || tmux new-session -d -s ffx \"source /root/xsenv.sh; bash /root/vast_ffx.sh > /root/vast_ffx.out 2>&1\"'"
vastai create instance $OFFER --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel --disk ${DISK:-150} --ssh --direct --label "$TAG" \
  --env "-e R2_ACCESS_KEY=$AK -e R2_SECRET=$SK -e R2_ENDPOINT=$EP -e FPA_TAG=$TAG -e FPA_SC=$SC -e FPA_COMMIT=$H ${FFX_ENV:-}" --onstart-cmd "$ONS" --raw
