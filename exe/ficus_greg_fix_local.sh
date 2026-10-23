#!/bin/bash
# ficus_greg_fix 로컬 감시·후처리 (tmux gregfix). SSH 안 씀: R2 마운트에 표식이 보이면 그림·영상·outputs.md·알림.
#   표식 GREGFIX_CLUSTER_DONE -> 후처리, GREGFIX_FAIL -> 알림만
set -u
M=/home/wgsong/workspace/execute/anchorflow/master
B=/home/wgsong/workspace/result/anchorflow/lbours/ficus_v0_greg_fix; RES=/home/wgsong/workspace/result/anchorflow
LOG=/tmp/gregfix_local.log
START=/tmp/gregfix_local.start; touch $START   # 이 시각 전의 실패 표식(이전 시도)은 무시
while true; do
  if [ -n "$(find $B -maxdepth 1 -name GREGFIX_FAIL -newer $START 2>/dev/null)" ]; then
    notify "ficus 로컬 변형 수정판 실패" "$(tail -1 $B/GREGFIX_FAIL)" --priority high --session anchorflow; exit 1; fi
  [ -f $B/GREGFIX_CLUSTER_DONE ] && break
  sleep 300
done
sleep 120   # 마지막 동기화 여유
cd $M
python3 exe/ficus_greg_fix_plot.py --ana $B/ana --out_png $RES/ovnet/ficus_greg_fix.png --out_md $RES/ovnet/ficus_greg_fix.md > $LOG 2>&1 || {
  notify "ficus 로컬 변형 그림 실패" "$(tail -2 $LOG)" --priority high --session anchorflow; exit 1; }
bash exe/ficus_greg_fix_vid.sh >> $LOG 2>&1 || { notify "ficus 로컬 변형 영상 실패" "$(tail -2 $LOG)" --priority high --session anchorflow; exit 1; }
ERR=$(cat $B/run/err_ov.txt 2>/dev/null)
{
  echo; echo "## $(date +%F) ficus 로컬 변형 버그 수정판 (--gregory, 고친 skin.py) 출력만 최적화 — ficus_v0_greg 다시 돌리기"
  echo "- Path: ~/workspace/result/anchorflow/ovnet/ficus_greg_fix.png (탄성·운동 에너지, 무게중심·줄기 끝 x, 반경 분포, 로컬/글로벌 ΔE), ficus_greg_fix.md (진동 표),"
  echo "  ~/workspace/result/anchorflow/v20261009/ficus_greg_fix.mp4 (3DGS 정답 | 선형 | 이전 Gregory | 수정 Gregory, 프레임 번호 빨간색), ficus_greg_fix_psi.mp4 (탄성 색, 공통 고정 범위 log10(ψ+1e-3) -3.000 ~ 4.933)"
  echo "- R2 원본: r2:storage/result/anchorflow/lbours/ficus_v0_greg_fix/ (run/: roll_ov.pt, h5_ov/, v_ov.log, inv_ov.jsonl, psi/, ov_ficus.mp4; ana/: osc.json, energy/, nodes_stats.json, split_el/, locked/; smoke/ 3 프레임 시험; cmdline.txt, unit_*.log)"
  echo "- Commit: 롤아웃 f599abb (브랜치 gregfix-c95f79c = c95f79c + 고친 skin.py 만), 분석 69df168 (master); 클러스터 /home/dkta/work/ficus_greg_fix.sh"
  echo "- Config: ficus_v0_greg (c95f79c) 와 명령줄 같음 (cmdline.txt 자동 대조 '같다'); 바뀐 것은 로컬 변형 식뿐: λ' = λ + W(ρ) − W(ρ_ref), x' = Σλ'(X+U) (옛: du = ΣW U)"
  echo "- 오차: $ERR (이전 Gregory 1.480%, 선형 1.788%)"
  echo "- 진동·에너지 요약:"; sed 's/^/  /' $RES/ovnet/ficus_greg_fix.md
  echo "- 잠긴 에너지 표: lbours/ficus_v0_greg_fix/ana/locked/ficus_greg_fix_locked.md"
} >> $RES/outputs.md
notify "ficus 로컬 변형 수정판 끝" "오차 $ERR, 그림 ovnet/ficus_greg_fix.png, 영상 v20261009/ficus_greg_fix.mp4" --tags white_check_mark --session anchorflow
echo GREGFIX_LOCAL_DONE >> $LOG
