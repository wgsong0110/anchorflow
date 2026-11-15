#!/bin/bash
# 절 CM185: GF 장면 우리 판 (CM184 수정 + CM185 칸 상한) vast 인스턴스 하나 = 카드 하나에 장면 차례로 (단독). 사용: bash exe/gfs185_launch.sh <offer> <태그> '<이름:모델:설정:프레임:추가,...>' [XS_DROP] [XS_CAUTO]
#  R2 키는 로컬 rclone 설정에서 읽어 --env 로만 넘김 (파일에 안 씀). 코드 = R2 lbours/gfscenes/code_cm185 (af_code.tar + exe/ + vast_gfs.sh)
set -e
OFFER=$1; TAG=$2; SC=$3; DROP=${4:---pdiag+--qjd+--fhash+--hgt+--bar_cnt}; CA=${5:-40000}
AK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['access_key_id'])")
SK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['secret_access_key'])")
EP=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['endpoint'])")
CODE=r2:storage/result/anchorflow/lbours/gfscenes/code_cm185
ONS="bash -c '(command -v unzip >/dev/null && command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux curl git unzip) >/dev/null 2>&1); command -v rclone >/dev/null || (curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/); mkdir -p /root/.config/rclone; printf \"[r2]\\ntype = s3\\nprovider = Cloudflare\\naccess_key_id = %s\\nsecret_access_key = %s\\nendpoint = %s\\nacl = private\\nno_check_bucket = true\\n\" \"\$R2_ACCESS_KEY\" \"\$R2_SECRET\" \"\$R2_ENDPOINT\" > /root/.config/rclone/rclone.conf; rclone copyto $CODE/vast_gfs.sh /root/vast_gfs.sh; env | grep -E \"^(R2_|XS_|CONTAINER_)\" | sed \"s/^/export /;s/=/=\\\"/;s/\$/\\\"/\" > /root/xsenv.sh; tmux has-session -t gfs 2>/dev/null || tmux new-session -d -s gfs \"source /root/xsenv.sh; bash /root/vast_gfs.sh > /root/vast_gfs.out 2>&1\"'"
vastai create instance $OFFER --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel --disk 150 --ssh --direct --label "$TAG" \
  --env "-e R2_ACCESS_KEY=$AK -e R2_SECRET=$SK -e R2_ENDPOINT=$EP -e XS_TAG=$TAG -e XS_STAGE=ours -e XS_HF=0 -e XS_CODE=$CODE -e XS_SCENES=$SC -e XS_DROP=$DROP -e XS_CAUTO=$CA" --onstart-cmd "$ONS" --raw
