#!/bin/bash
# 절 CM192: 정답 렌더 대비 PSNR·SSIM·LPIPS 가 빠진 지표 폴더 다시 (지표 폴더의 png 와 R2 phys/mpm_<장면>/png 로 ffx_vis.py --ref; 다른 시각 지표도 같은 코드로 다시 계산됨).
#   대기열 R2 lbours/final_fix/queue/<FX_LABEL>.txt 한 줄 = "<장면> <지표 폴더 이름>", 끝 표식 R2 phys/<폴더>/VISREF_DONE (잠금 VISREF_LOCK)
export PATH=/home/dkta/.conda/envs/af/bin:/home/dkta/bin:$PATH
FX=r2:storage/result/anchorflow/lbours/final_fix; RP=$FX/phys; Q=$FX/queue/${FX_LABEL:?}.txt; X=/workspace/vr; mkdir -p $X /root/ref; WID=${WID:-0}
PY=/home/dkta/.conda/envs/af/bin/python
rclone copyto $Q /root/vr_$WID.txt -q
while read -r T J; do
  [ -z "${J:-}" ] && continue
  rclone lsf $RP/$J/VISREF_DONE 2>/dev/null | grep -q . && continue
  rclone lsf $RP/$J/VISREF_LOCK 2>/dev/null | grep -q . && continue
  mkdir $X/$J 2>/dev/null || continue
  echo "$(hostname) w$WID $(date)" | rclone rcat $RP/$J/VISREF_LOCK -q
  [ -d /root/ref/$T ] || { mkdir -p /root/ref/$T.tmp$WID && rclone copy $RP/mpm_$T/png /root/ref/$T.tmp$WID -q --transfers 16 && { mv /root/ref/$T.tmp$WID /root/ref/$T 2>/dev/null || true; }; }
  rclone copy $RP/$J/png $X/$J/png -q --transfers 16
  rclone copyto $RP/$J/vis.json $X/$J/vis_noref.json -q
  $PY -u exe/ffx_vis.py --png $X/$J/png --ref /root/ref/$T --out $X/$J/vis.json > $X/$J/vis.log 2>&1
  if grep -q '"psnr"' $X/$J/vis.json 2>/dev/null; then
    ( rclone copyto $X/$J/vis_noref.json $RP/$J/vis_noref.json -q; rclone copyto $X/$J/vis.json $RP/$J/vis.json -q; rclone copyto $X/$J/vis.log $RP/$J/vis_ref.log -q
      echo "$(date)" | rclone rcat $RP/$J/VISREF_DONE -q; rm -rf $X/$J/png ) &
    echo "[vr w$WID] $J $(grep -o '"psnr": [0-9.]*' $X/$J/vis.json)"
  else echo "[vr w$WID] $J 실패 $(tail -2 $X/$J/vis.log)"; fi
done < /root/vr_$WID.txt
wait
