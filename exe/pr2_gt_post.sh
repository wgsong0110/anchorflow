#!/bin/bash
# [paper_remeasure2] pr2_gt.sh 후처리 (h5 는 <out>/simulation_ply/ 아래라 pr2_gt.sh 의 변환 조건이 맞지 않았음): 변환 + 이웃 이탈 측정 + R2 업로드. cwd = 코드 폴더, 환경 = penv.sh
set -u
source /root/penv.sh
R=r2:storage/result/anchorflow; RX=$R/lbours/paper_remeasure2; W=/home/dkta/work
export PATH=/home/dkta/.conda/envs/af/bin:$PATH HOME=/root
for k in vasedeck plane pillow2sofa; do
  O=$W/gtnew/$k; n=$(ls $O/simulation_ply/sim_*.h5 2>/dev/null | wc -l)
  echo "[후처리] $k h5 $n 개"
  [ "$n" -gt 1 ] || continue
  python -u exe/pg_h5_to_pt.py --h5_dir $O --config $W/cfgnew/${k}_config.json --out_pt $W/gtnew/$k.pt --out_dir $W/gtnew/gt_$k > $W/gtnew/${k}_conv.log 2>&1; echo "변환 rc=$? $(tail -1 $W/gtnew/${k}_conv.log | cut -c1-200)"
  python -u exe/measure_gf_tearing.py --h5_dir $O --tag $k --out $W/gtnew/tear_$k > $W/gtnew/${k}_tear.log 2>&1; echo "이탈 rc=$?"
  rclone copy $W/gtnew $RX/scenes/$k/ -q --include "$k.pt" --include "gt_$k/**" --include "${k}_*.log" --include "tear_$k/**" --include "${k}.log" --transfers 8
  rclone copy $O $RX/scenes/$k/h5 -q --transfers 8 && echo ok | rclone rcat $RX/scenes/$k/DONE
  echo "[후처리] $k 업로드 끝"
done
echo POST_END
