#!/bin/bash
# 절 AJ 후속 후처리 (CPU 만): 재실행 기록 dc_wm_vh2 를 최신 요약 스크립트로 다시 요약 (최고 속력 분포 포함) + 두 재실행 (vh, vh2) 의 속력 기록 대조 -> R2
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1 OMP_NUM_THREADS=8; set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
B=/tmp/dkta/lbours/fem_dc; RR=r2:storage/result/anchorflow/lbours/fem_dc
echo "[커밋] $(git log -1 --oneline) $(date '+%F %T')"
until [ -f $B/dc_wm_vh2/VH_DONE ]; do sleep 60; done
python -u exe/wm_divnode_stats.py --dump $B/dc_wm_vh2/divhist/f003.pt --out $B/dc_wm_vh2/divnode_full.npz < /dev/null > $B/dc_wm_vh2/divnode_full.log 2>&1; tail -2 $B/dc_wm_vh2/divnode_full.log | cut -c1-1500
python -u - <<'PY' > $B/dc_wm_vh2/vh_vs_vh2.txt 2>&1
import torch
a = torch.load("/tmp/dkta/lbours/fem_dc/dc_wm_vh/divhist/f003.pt", map_location="cpu", weights_only=False)
b = torch.load("/tmp/dkta/lbours/fem_dc/dc_wm_vh2/divhist/f003.pt", map_location="cpu", weights_only=False)
d = (a["speed"] - b["speed"]).abs()
print("속력 최대 차", float(d.max()), "상대 중앙", float((d / b["speed"].clamp_min(1e-6)).median()))
print("발산 서브스텝 같은 노드", int((a["abs"] == b["abs"]).sum()), "/", a["abs"].numel(), "넘은 노드 수", int((a["abs"] >= 0).sum()), int((b["abs"] >= 0).sum()))
print("처음 넘은 서브스텝 (0 부터) 전체", int(a["abs"][a["abs"] >= 0].min()), int(b["abs"][b["abs"] >= 0].min()))
PY
cat $B/dc_wm_vh2/vh_vs_vh2.txt
rclone copy $B/dc_wm_vh2 $RR/dc_wm_vh2 -q && echo "[R2] dc_wm_vh2"
echo "POST_DONE $(date +%T)"
