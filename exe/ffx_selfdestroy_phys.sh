#!/bin/bash
# 절 CM192: 이 인스턴스의 남은 일 (nd2 지표 작업) 이 끝나면 R2 동기화·확인 뒤 스스로 destroy (키는 컨테이너 환경 CONTAINER_API_KEY 그대로, 파일에 안 남김)
source /root/fxpenv.sh; export PATH=/home/dkta/bin:/opt/conda/bin:$PATH
RP=r2:storage/result/anchorflow/lbours/final_fix/phys; MK=r2:storage/result/anchorflow/lbours/final_fix/marks/final_fix_phys
while :; do ok=1; for j in $(cut -d" " -f3 /root/nd2_jobs.txt); do [ -f /workspace/fxp/$j/DONE ] || ok=0; done; [ $ok = 1 ] && break; sleep 60; done
sleep 30
rclone copy /workspace/fxp $RP -q --transfers 8 --exclude "*/h5/**" --exclude "*/pred.pt" --exclude "*.claim/**"
for f in /root/nd2_phys*.log /root/vr_it1.log; do rclone copyto $f $MK.$(basename $f) -q; done
if rclone check /workspace/fxp $RP --one-way --size-only -q --exclude "*/h5/**" --exclude "*/pred.pt" --exclude "*.claim/**" 2>/root/check.txt; then
  echo "$(date) R2 확인 차이 0 -> destroy" | rclone rcat $MK.SELF_DESTROY
  vastai destroy instance ${CONTAINER_ID:-0} -y --api-key ${CONTAINER_API_KEY:-x}
else
  echo "$(date) R2 확인 차이 있음 -- destroy 안 함" | rclone rcat $MK.SELF_DESTROY_FAIL
fi
