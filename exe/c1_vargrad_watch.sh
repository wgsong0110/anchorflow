#!/bin/bash
# ficus C¹ 변수 기울기(AF_VARGRAD) 클러스터 실행 감시 + 끝나면 정리 (로컬 tmux). 표식: /tmp/c1vargrad_local.log 에 C1VARGRAD_DONE / C1VARGRAD_FAIL
L=/tmp/c1vargrad_local.log; RD=/home/dkta/work/lbours/ficus_c1/vargrad; LOCAL=/tmp/c1vg/full
REPO=/home/wgsong/workspace/execute/anchorflow/master; RES=/home/wgsong/workspace/result/anchorflow
TEX=/home/wgsong/workspace/writing/anchorflow/c1_results.tex
echo "[감시 시작] $(date)" >> $L
t0=$(date +%s)
while true; do
  st=$(ssh -o ConnectTimeout=20 dkta@147.46.121.39 "if [ -f $RD/C1VARGRAD_DONE ]; then echo DONE; elif [ -f $RD/C1VARGRAD_FAIL ]; then echo FAIL; else echo RUN \$(cat $RD/*/vargrad.jsonl 2>/dev/null | wc -l); fi" 2>/dev/null)
  echo "[$(date +%T)] $st" >> $L
  case "$st" in
    DONE*) break;;
    FAIL*) echo C1VARGRAD_FAIL >> $L; exit 1;;
  esac
  [ $(( $(date +%s) - t0 )) -gt 28800 ] && { echo "[시간 초과 8 h]" >> $L; echo C1VARGRAD_FAIL >> $L; exit 1; }
  sleep 600
done
mkdir -p $LOCAL
for n in c1 bary basenone; do mkdir -p $LOCAL/$n; ssh dkta@147.46.121.39 "cat $RD/$n/vargrad.jsonl" > $LOCAL/$n/vargrad.jsonl; done
python3 $REPO/exe/c1_vargrad_report.py --vg $LOCAL --inv $RES/lbours/ficus_c1/N1/inv_ov.jsonl \
  --png $RES/ovnet/ficus_c1_vargrad.png --md $RES/ovnet/ficus_c1_vargrad.md --tex $LOCAL/section.tex >> $L 2>&1 || { echo C1VARGRAD_FAIL >> $L; exit 1; }
mkdir -p $RES/lbours/ficus_c1/vargrad
cp $RES/ovnet/ficus_c1_vargrad.png $RES/ovnet/ficus_c1_vargrad.md $LOCAL/section.tex $RES/lbours/ficus_c1/vargrad/ 2>>$L
# 클러스터에서 R2 로 마지막 통째 동기화는 잡 스크립트가 했다. 여기서는 그림·표만 더한다.
python3 - $TEX $LOCAL/section.tex <<'PY' >> $L 2>&1
import sys
p, s = sys.argv[1], open(sys.argv[2]).read()
t = open(p).read()
if "최적화 변수 기준 기울기 (2026-10-21)" in t:
    print("[tex] 이미 있음 -- 건너뜀")
else:
    i = t.rindex(r"\end{document}")
    open(p, "w").write(t[:i] + s + "\n" + t[i:]); print("[tex] 절 추가", p)
PY
C=$(cd $REPO && git log -1 --format=%h -- exe/train_deform.py)
cat >> $RES/outputs.md <<MD

## 2026-10-21 ficus C¹ N1 최적화 변수 θ 기준 기울기 (AF_VARGRAD)
- Path: r2:storage/result/anchorflow/lbours/ficus_c1/vargrad/ (c1 = C¹ N1 덤프 역변환 125 스텝, bary = 선형 ficus_v0 덤프 역변환 + 다시 풀어 종료 통계, basenone = c1_base none 첫 스텝 500 회 다시 풀기; 각 vargrad.jsonl·vg_t*.pt(g·θ 벡터·CG 이력)·v.log), 그림·표 ~/workspace/result/anchorflow/ovnet/ficus_c1_vargrad.{png,md}
- Commit: $C (train_deform AF_VARGRAD, wf.PL_DEDUP 측정용), 클러스터 /home/dkta/work/ficus_c1vargrad.sh
- Config: ficus_c1.sh 와 같은 명령줄 (--interp c1wf --c1_barrier 1.0 --c1_jhat 0.3, c1_base pl), AF_OV_NODE_TGT 로 상태를 덤프에 맞춤, CG 100
- Files: c1_results.tex 절 "ficus C¹ N1 최적화 변수 기준 기울기"
MD
echo C1VARGRAD_DONE >> $L
