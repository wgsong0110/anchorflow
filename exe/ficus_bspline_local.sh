#!/bin/bash
# ficus_bspline 로컬 감시·후처리 (tmux bsplocal). SSH 안 씀: R2 마운트에 표식이 보이면 그림·표·영상·outputs.md·알림.
#   표식 BSP_CLUSTER_DONE -> 후처리, BSP_FAIL -> 알림만
set -u
M=/home/wgsong/workspace/execute/anchorflow/master
B=/home/wgsong/workspace/result/anchorflow/lbours/ficus_v0_bspline; RES=/home/wgsong/workspace/result/anchorflow
LOG=/tmp/bsp_local.log
START=/tmp/bsp_local.start; touch $START
while true; do
  if [ -n "$(find $B -maxdepth 1 -name BSP_FAIL -newer $START 2>/dev/null)" ]; then
    notify "ficus B-스플라인 실패" "$(tail -1 $B/BSP_FAIL)" --priority high --session anchorflow; exit 1; fi
  [ -f $B/BSP_CLUSTER_DONE ] && break
  sleep 300
done
sleep 120
cd $M
python3 exe/ficus_bspline_plot.py --ana $B/ana --old_quick $RES/lbours/ficus_v0_localrot/ana/osc_quick.json \
  --out_png $RES/ovnet/ficus_bspline.png --out_md $RES/ovnet/ficus_bspline.md > $LOG 2>&1 || {
  notify "ficus B-스플라인 그림 실패" "$(tail -2 $LOG)" --priority high --session anchorflow; exit 1; }
bash exe/ficus_bspline_vid.sh >> $LOG 2>&1 || { notify "ficus B-스플라인 영상 실패" "$(tail -2 $LOG)" --priority high --session anchorflow; exit 1; }
EA=$(cat $B/A/err_ov.txt 2>/dev/null); EB=$(cat $B/B/err_ov.txt 2>/dev/null)
{
  echo; echo "## $(date +%F) ficus 텐서곱 B-스플라인 FFD 출력만 최적화 (A --interp bspline2 --cell_barrier 1.0, B --interp bspline3)"
  echo "- Path: ~/workspace/result/anchorflow/ovnet/ficus_bspline.png (에너지·ψ·진동·줄기 끝·위치 오차·det F·잠긴 에너지·B 상한), ficus_bspline.md (표),"
  echo "  ~/workspace/result/anchorflow/v20261009/ficus_bspline.mp4 (3DGS 2×3: 정답 | 선형 | 수정 Gregory / 회전판 | A | B, 프레임 번호 빨간색), ficus_bspline_psi.mp4 (탄성 색, 공통 고정 범위 log10(ψ+1e-3) -3.000 ~ 4.933)"
  echo "- R2 원본: r2:storage/result/anchorflow/lbours/ficus_v0_bspline/ (A/, B/: roll_ov.pt, h5_ov/, v_ov.log, inv_ov.jsonl, psi/, ov_ficus.mp4, cmd.txt; smoke/ 3 프레임 시험; ana/: osc.json, osc_quick.json, energy/, detstats.json, locked/; cmdline.txt, unit_6e27a7e.log)"
  echo "- Commit: 6e27a7e (master; 구현 36104b7), 클러스터 /home/dkta/work/ficus_bspline.sh"
  echo "- Config: ficus_v0 명령줄([BASE]·[OV]) 그대로 + (A) --interp bspline2 --cell_barrier 1.0 (Ĵ 0.3, 연장 linear, 표본점 8/조각) / (B) --interp bspline3 (축 이웃 변위 차 상한 0.48Δ, Δ = 0.04)"
  echo "- 오차: A $EA, B $EB (선형 1.788%)"
  echo "- 요약:"; sed 's/^/  /' $RES/ovnet/ficus_bspline.md
} >> $RES/outputs.md
notify "ficus B-스플라인 끝" "A $EA, B $EB; ovnet/ficus_bspline.png, v20261009/ficus_bspline.mp4" --tags white_check_mark --session anchorflow
echo BSP_LOCAL_DONE >> $LOG
