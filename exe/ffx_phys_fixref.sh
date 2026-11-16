#!/bin/bash
# [final_fix_phys] 정답 렌더 대비 지표가 빠진 작업 고침: 두 번째 인스턴스가 R2 에 끝난 정답 MPM 작업 (mpm_<장면>) 을 건너뛰어 그 PNG 가 로컬에 없었음 -> ffx_vis --ref 가 빈 폴더.
#   작업 이름들을 인자로; PNG 는 R2 lbours/final_fix/phys/<이름>/png 에서 받음. 코드 폴더에서: bash exe/ffx_phys_fixref.sh <이름>...
set -u
RP=r2:storage/result/anchorflow/lbours/final_fix/phys; Y=/workspace/fxref; PY=/home/dkta/.conda/envs/af/bin/python; mkdir -p $Y
for J in "$@"; do
  T=$(echo $J | sed -E 's/^ours_//; s/_[^_]+$//')
  [ -d $Y/mpm_$T/png ] || rclone copy $RP/mpm_$T/png $Y/mpm_$T/png -q --transfers 16
  mkdir -p $Y/$J; rclone copy $RP/$J/png $Y/$J/png -q --transfers 16; rclone copyto $RP/$J/vis.json $Y/$J/vis.json -q
  $PY -u exe/ffx_vis.py --png $Y/$J/png --ref $Y/mpm_$T/png --out $Y/$J/vis.json --ref_only > $Y/$J/vis_ref.log 2>&1
  echo "[$J] $(tail -n 1 $Y/$J/vis_ref.log | cut -c1-200)"
  grep -q psnr $Y/$J/vis.json && rclone copyto $Y/$J/vis.json $RP/$J/vis.json -q && rclone copyto $Y/$J/vis_ref.log $RP/$J/vis_ref.log -q
done
echo FIXREF_DONE
