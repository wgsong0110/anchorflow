#!/bin/bash
# [final_fix_phys] 인스턴스 하나 (라벨 final_fix_phys). 사용: bash exe/ffx_phys_launch.sh <offer> <FB_COMMIT 베이스라인 실행기 tar> <FX_COMMIT 이 판 tar> ["FX_ONLY 작업들"]
#  R2 키는 로컬 rclone 설정에서 읽어 --env 로만 넘김 (파일에 안 씀; exe/ffx_launch.sh 와 같은 방식).
set -e
OFFER=$1; FB=$2; FXC=$3; ONLY=${4:-}; CODE=r2:storage/result/anchorflow/lbours/final_fix/code
AK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['access_key_id'])")
SK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['secret_access_key'])")
EP=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['endpoint'])")
ONS="bash -c '(command -v unzip >/dev/null && command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux curl git unzip) >/dev/null 2>&1); command -v rclone >/dev/null || (curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/); mkdir -p /root/.config/rclone; printf \"[r2]\\ntype = s3\\nprovider = Cloudflare\\naccess_key_id = %s\\nsecret_access_key = %s\\nendpoint = %s\\nacl = private\\nno_check_bucket = true\\n\" \"\$R2_ACCESS_KEY\" \"\$R2_SECRET\" \"\$R2_ENDPOINT\" > /root/.config/rclone/rclone.conf; rclone cat $CODE/ffxphys_$FXC.tar | tar -xO exe/ffx_phys_boot.sh > /root/fxp_boot.sh; env | grep -E \"^(R2_|FB_|FX_|CONTAINER_)\" | sed \"s/^/export /;s/=/=\\\"/;s/\$/\\\"/\" > /root/fxpenv.sh; tmux has-session -t fxp 2>/dev/null || tmux new-session -d -s fxp \"source /root/fxpenv.sh; bash /root/fxp_boot.sh > /root/fxp_boot.out 2>&1\"'"
EO=""
vastai create instance $OFFER --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel --disk ${DISK:-400} --ssh --direct --label final_fix_phys \
  --env "-e R2_ACCESS_KEY=$AK -e R2_SECRET=$SK -e R2_ENDPOINT=$EP -e FB_COMMIT=$FB -e FX_COMMIT=$FXC $EO" --onstart-cmd "$ONS" --raw
