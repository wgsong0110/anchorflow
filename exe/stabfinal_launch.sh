#!/bin/bash
# [stabfinal] vast 인스턴스 하나 띄우기 (새로) 또는 기존 인스턴스에 ssh 로 부팅 걸기.
#  새로:  bash exe/stabfinal_launch.sh new <offer id> <태그> '<씬:단계+씬:단계>' <커밋> [동시 판 수]
#  기존:  bash exe/stabfinal_launch.sh ssh <인스턴스 id> <태그> '<씬:단계+씬:단계>' <커밋> [동시 판 수]   (tmux sf 로 띄움; 키는 기존 rclone 설정 또는 env 로)
#  R2 키는 로컬 rclone 설정에서 읽어 --env / ssh 명령 인자로만 넘김 (파일에 안 씀). 부팅 스크립트 = R2 lbours/stabfinal/code/boot_<커밋>.sh
set -e
MODE=$1; ID=$2; TAG=$3; JOBS=$4; C=$5; PAR=${6:-4}
AK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['access_key_id'])")
SK=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['secret_access_key'])")
EP=$(rclone config dump | python3 -c "import sys,json;print(json.load(sys.stdin)['r2']['endpoint'])")
if [ "$MODE" = new ]; then
ONS="bash -c '(command -v unzip >/dev/null && command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux curl git unzip) >/dev/null 2>&1); command -v rclone >/dev/null || (curl -sL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/rc.zip && unzip -q -o /tmp/rc.zip -d /tmp && cp /tmp/rclone-*/rclone /usr/local/bin/); mkdir -p /root/.config/rclone; printf \"[r2]\\ntype = s3\\nprovider = Cloudflare\\naccess_key_id = %s\\nsecret_access_key = %s\\nendpoint = %s\\nacl = private\\nno_check_bucket = true\\n\" \"\$R2_ACCESS_KEY\" \"\$R2_SECRET\" \"\$R2_ENDPOINT\" > /root/.config/rclone/rclone.conf; rclone copyto r2:storage/result/anchorflow/lbours/stabfinal/code/boot_$C.sh /root/boot_sf.sh; env | grep -E \"^(R2_|SF_|CONTAINER_)\" | sed \"s/^/export /\" > /root/sfenv.sh; tmux has-session -t sf 2>/dev/null || tmux new-session -d -s sf \"source /root/sfenv.sh; bash /root/boot_sf.sh > /root/boot_sf.out 2>&1\"'"
vastai create instance $ID --image pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel --disk 80 --ssh --direct --label "final_stab_$TAG" \
  --env "-e R2_ACCESS_KEY=$AK -e R2_SECRET=$SK -e R2_ENDPOINT=$EP -e SF_COMMIT=$C -e SF_JOBS=$JOBS -e SF_TAG=$TAG -e SF_PAR=$PAR" --onstart-cmd "$ONS" --raw
else
  vastai label instance $ID "final_stab_$TAG" >/dev/null
  read HOST PORT < <(vastai show instances-v1 --raw | python3 -c "
import sys,json
d=json.load(sys.stdin); L=d if isinstance(d,list) else d.get('instances',d)
for i in L:
    if str(i['id'])=='$ID': print(i['ssh_host'], i['ssh_port'])")
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $PORT root@$HOST "mkdir -p /root/.config/rclone; [ -s /root/.config/rclone/rclone.conf ] || printf '[r2]\ntype = s3\nprovider = Cloudflare\naccess_key_id = %s\nsecret_access_key = %s\nendpoint = %s\nacl = private\nno_check_bucket = true\n' '$AK' '$SK' '$EP' > /root/.config/rclone/rclone.conf; rclone copyto r2:storage/result/anchorflow/lbours/stabfinal/code/boot_$C.sh /root/boot_sf.sh && (tr '\0' '\n' < /proc/1/environ | grep -E '^CONTAINER_(ID|API_KEY)=' | sed 's/^/export /'; echo 'export SF_COMMIT=$C SF_JOBS=\"$JOBS\" SF_TAG=$TAG SF_PAR=$PAR') > /root/sfenv.sh && (tmux has-session -t sf 2>/dev/null || tmux new-session -d -s sf 'source /root/sfenv.sh; bash /root/boot_sf.sh > /root/boot_sf.out 2>&1') && echo launched"
fi
