#!/bin/bash
# [final_fix_mpmbl] 인스턴스 하나 (라벨 final_fix_mpmbl_<방법>). 사용: bash exe/mpmbl_launch.sh <offer> <커밋> <ipg|omni> ["장면들"]
#  R2 키는 로컬 rclone 설정에서 읽어 --env 로만 넘김 (파일에 안 씀). 부팅 = R2 lbours/final_fix/bl_mpm/code/mpmbl_<커밋>.tar 의 exe/mpmbl_boot.sh
set -e
OFFER=$1; C=$2; M=$3; SC=${4:-"L000 bread pillow2sofa wolf"}; CODE=r2:storage/result/anchorflow/lbours/final_fix/bl_mpm/code
AK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['access_key_id'])")
SK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['secret_access_key'])")
EP=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['endpoint'])")
ONS="bash -c '(command -v unzip >/dev/null && command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux curl git unzip) >/dev/null 2>&1); command -v rclone >/dev/null || (curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/); mkdir -p /root/.config/rclone; printf \"[r2]\\ntype = s3\\nprovider = Cloudflare\\naccess_key_id = %s\\nsecret_access_key = %s\\nendpoint = %s\\nacl = private\\nno_check_bucket = true\\n\" \"\$R2_ACCESS_KEY\" \"\$R2_SECRET\" \"\$R2_ENDPOINT\" > /root/.config/rclone/rclone.conf; rclone cat $CODE/mpmbl_$C.tar | tar -xO exe/mpmbl_boot.sh > /root/mb_boot.sh; env | grep -E \"^(R2_|MB_|CONTAINER_)\" | sed \"s/^/export /;s/=/=\\\"/;s/\$/\\\"/\" > /root/mbenv.sh; tmux has-session -t mb 2>/dev/null || tmux new-session -d -s mb \"source /root/mbenv.sh; bash /root/mb_boot.sh > /root/mb_boot.out 2>&1\"'"
vastai create instance $OFFER --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel --disk ${DISK:-200} --ssh --direct --label final_fix_mpmbl_$M \
  --env "-e R2_ACCESS_KEY=$AK -e R2_SECRET=$SK -e R2_ENDPOINT=$EP -e MB_COMMIT=$C -e MB_METHOD=$M -e MB_TAG=final_fix_mpmbl_$M -e MB_SCENES=\"$SC\"" --onstart-cmd "$ONS" --raw
