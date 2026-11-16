#!/bin/bash
# [blx] (exe/fpabl_launch.sh 복사) 인스턴스 하나 띄우기 (exe/nshape_launch.sh 와 같은 방식: R2 키는 로컬 rclone 설정에서 읽어 --env 로만, 파일에 안 씀).
#  사용: bash exe/blx_launch.sh <offer> <라벨 final_blx_*> <커밋> <태그> <작업 json> <prep|grid> [FB_PREP_FROM] [hold]
set -e
OFFER=$1; LBL=$2; H=$3; TAG=$4; JOBS=$5; MODE=$6; PF=${7:-}; HOLD=${8:-}
case $LBL in final_blx_*) ;; *) echo "라벨은 final_blx_ 로 시작"; exit 1;; esac
OX=r2:storage/result/anchorflow/lbours/final_pa/blx
AK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['access_key_id'])")
SK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['secret_access_key'])")
EP=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['endpoint'])")
ONS="bash -c '(command -v unzip >/dev/null && command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux curl git unzip) >/dev/null 2>&1); command -v rclone >/dev/null || (curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/); mkdir -p /root/.config/rclone; printf \"[r2]\\ntype = s3\\nprovider = Cloudflare\\naccess_key_id = %s\\nsecret_access_key = %s\\nendpoint = %s\\nacl = private\\nno_check_bucket = true\\n\" \"\$R2_ACCESS_KEY\" \"\$R2_SECRET\" \"\$R2_ENDPOINT\" > /root/.config/rclone/rclone.conf; rclone cat $OX/code/blx_$H.tar | tar -xO exe/vast_blx_boot.sh > /root/vast_fpabl_boot.sh; env | grep -E \"^(R2_|FB_|CONTAINER_)\" | sed \"s/^/export /;s/=/=\\\"/;s/\$/\\\"/\" > /root/fbenv.sh; tmux has-session -t fb 2>/dev/null || tmux new-session -d -s fb \"source /root/fbenv.sh; bash /root/vast_fpabl_boot.sh > /root/vast_fpabl_boot.out 2>&1\"'"
EH=""; [ -n "$HOLD" ] && EH="-e FB_HOLD=1"
EPF=""; [ -n "$PF" ] && EPF="-e FB_PREP_FROM=$PF"; [ -n "${FB_VIS_FROM:-}" ] && EPF="$EPF -e FB_VIS_FROM=$FB_VIS_FROM"
vastai create instance $OFFER --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel --disk ${DISK:-200} --ssh --direct --label "$LBL" \
  --env "-e R2_ACCESS_KEY=$AK -e R2_SECRET=$SK -e R2_ENDPOINT=$EP -e FB_COMMIT=$H -e FB_TAG=$TAG -e FB_JOBS=$JOBS -e FB_MODE=$MODE $EPF $EH" --onstart-cmd "$ONS" --raw
