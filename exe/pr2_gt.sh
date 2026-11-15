#!/bin/bash
# [paper_remeasure2] PhysGaussian 논문 장면 (vasedeck · plane · pillow2sofa) 정답 궤적: 공식 config 그대로, i-physgaussian 의 gs_simulation.py --output_h5.
# 끝나면 데이터 .pt (pg_h5_to_pt) + 정답 폴더를 R2 lbours/paper_remeasure2/scenes/<씬>/ 에. (vast_pr2_boot.sh 가 PR_MODE=gt 일 때 부름; cwd = 코드 폴더)
set -u
R=r2:storage/result/anchorflow; RX=$R/lbours/paper_remeasure2; A=$RX/assets; W=/home/dkta/work
mkdir -p $W/gtnew $W/cfgnew
# 미러 gs_simulation.py 의 두 결함: (1) 채우기 의존 (PyMCubes) 없음 (2) _af_os 가 try 안에서만 import 됨 -> 최상단 import
python -m pip install -q pymcubes 2>&1 | tail -1
grep -q "AF_PR2_IMP" $W/i-physgaussian/gs_simulation.py || sed -i '1i import os as _af_os, numpy as _af_np, torch as _af_t  # AF_PR2_IMP' $W/i-physgaussian/gs_simulation.py
for pair in vasedeck:vasedeck_whitebg-trained plane:plane-trained pillow2sofa:pillow2sofa_whitebg-trained; do
  k=${pair%%:*}; mdl=${pair##*:}; O=$W/gtnew/$k
  [ -d $W/pgmodel/$mdl ] || { rclone copyto $A/$mdl.zip /root/pr/$mdl.zip -q && unzip -q -o /root/pr/$mdl.zip -d $W/pgmodel; }
  rclone copyto $A/${k}_config.json $W/cfgnew/${k}_config.json -q
  [ -s $RX/scenes/$k/DONE ] && continue
  say "[정답] $k 시작 (모델 $(ls $W/pgmodel/$mdl | tr '\n' ' '))"
  ( cd $W/i-physgaussian && AF_PGFILL_NPY=$W/gtnew/fill_$k.npy WARP_CACHE_PATH=$W/wpc_$k PYTHONIOENCODING=utf-8 timeout 14400 python -u gs_simulation.py --model_path $W/pgmodel/$mdl --config $W/cfgnew/${k}_config.json --output_path $O --output_h5 > $W/gtnew/$k.log 2>&1 ); rc=$?
  say "[정답] $k rc=$rc h5 $(ls $O/sim_*.h5 2>/dev/null | wc -l) 개 $(du -sh $O 2>/dev/null | cut -f1) | $(tail -2 $W/gtnew/$k.log | tr '\n' ' ' | cut -c1-300)"
  rclone copy $W/gtnew/$k.log $RX/scenes/$k/ -q
  if ls $O/sim_*.h5 >/dev/null 2>&1; then
    python -u exe/pg_h5_to_pt.py --h5_dir $O --config $W/cfgnew/${k}_config.json --out_pt $W/gtnew/$k.pt --out_dir $W/gtnew/gt_$k > $W/gtnew/${k}_conv.log 2>&1; say "[변환] $k rc=$? $(tail -1 $W/gtnew/${k}_conv.log | cut -c1-200)"
    python -u exe/measure_gf_tearing.py --h5_dir $O --tag $k --out $W/gtnew/tear_$k > $W/gtnew/${k}_tear.log 2>&1
    rclone copy $W/gtnew $RX/scenes/$k/ -q --include "$k.pt" --include "gt_$k/**" --include "${k}_*.log" --include "tear_$k/**" --include "${k}.log" --transfers 8
    rclone copy $O $RX/scenes/$k/h5 -q --transfers 8 && echo ok | rclone rcat $RX/scenes/$k/DONE
  fi
done
say "[정답 끝]"
echo "$(date)" | rclone rcat $MK/GT_END
