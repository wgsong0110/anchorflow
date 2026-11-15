#!/bin/bash
# 절 CM186: 새 형상 장면 하나 = vast 인스턴스 하나 (A6000 1 장). 사용: bash exe/nshape_launch.sh <offer> <장면> <물체> <pg|gf> <코드 R2 경로>
#  R2 키는 로컬 rclone 설정에서 읽어 --env 로만 넘김 (파일에 안 씀). 라벨 nshape_<장면>.
set -e
OFFER=$1; SC=$2; OBJ=$3; SOLV=$4; CODE=$5; TAG=nshape_$SC
AK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['access_key_id'])")
SK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['secret_access_key'])")
EP=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['endpoint'])")
ONS="bash -c '(command -v unzip >/dev/null && command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux curl git unzip) >/dev/null 2>&1); command -v rclone >/dev/null || (curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/); mkdir -p /root/.config/rclone; printf \"[r2]\\ntype = s3\\nprovider = Cloudflare\\naccess_key_id = %s\\nsecret_access_key = %s\\nendpoint = %s\\nacl = private\\nno_check_bucket = true\\n\" \"\$R2_ACCESS_KEY\" \"\$R2_SECRET\" \"\$R2_ENDPOINT\" > /root/.config/rclone/rclone.conf; rclone copyto $CODE/${BOOT:-vast_nshape.sh} /root/vast_nshape.sh; env | grep -E \"^(R2_|XS_|CONTAINER_)\" | sed \"s/^/export /;s/=/=\\\"/;s/\$/\\\"/\" > /root/xsenv.sh; tmux has-session -t ns 2>/dev/null || tmux new-session -d -s ns \"source /root/xsenv.sh; bash /root/vast_nshape.sh > /root/vast_nshape.out 2>&1\"'"
vastai create instance $OFFER --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel --disk ${DISK:-120} --ssh --direct --label "$TAG" \
  --env "-e R2_ACCESS_KEY=$AK -e R2_SECRET=$SK -e R2_ENDPOINT=$EP -e XS_TAG=$TAG -e XS_SC=$SC -e XS_OBJ=$OBJ -e XS_SOLVER=$SOLV -e XS_CODE=$CODE" --onstart-cmd "$ONS" --raw
