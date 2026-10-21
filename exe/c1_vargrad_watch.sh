#!/bin/bash
# ficus 변수 기울기(AF_VARGRAD) 감시·정리 (로컬 tmux c1vargrad). 표식 /tmp/c1vargrad_local.log:
#   C1VARGRAD_BARY_DONE  선형(+C¹ none 첫 스텝) 그림·표 (버그판 C¹ 측정은 10-21 지시로 중지, 46 스텝 부분 결과만 남김)
#   C1VARGRAD_DONE       고친 코드 C¹ (lbours/ficus_c1fix/N1) 측정 포함 최종 (그림·표·tex·outputs.md)
#   C1VARGRAD_FAIL
# 순서: 선형 정리 -> /tmp/c1fix_local.log 에 C1FIX_DONE 기다림 -> 그 커밋으로 클러스터 측정(ficus_c1vargrad_fix.sh, job_run.sh) -> 최종 정리
L=/tmp/c1vargrad_local.log; LOCAL=/tmp/c1vg/full; FIXL=/tmp/c1fix_local.log
REPO=/home/wgsong/workspace/execute/anchorflow/master; RES=/home/wgsong/workspace/result/anchorflow
TEX=/home/wgsong/workspace/writing/anchorflow/c1_results.tex
RB=/home/dkta/work/lbours/ficus_c1/vargrad; RF=/home/dkta/work/lbours/ficus_c1fix
SSH="ssh -o ConnectTimeout=20 dkta@147.46.121.39"
log(){ echo "[$(date +%T)] $*" >> $L; }
die(){ log "$*"; echo C1VARGRAD_FAIL >> $L; exit 1; }
log "감시 v2 시작"
# ---- 1) 선형·none 정리
if ! grep -q C1VARGRAD_BARY_DONE $L 2>/dev/null; then
  for n in bary basenone; do mkdir -p $LOCAL/$n; $SSH "cat $RB/$n/vargrad.jsonl" > $LOCAL/$n/vargrad.jsonl || die "$n 가져오기 실패"; done
  [ $(wc -l < $LOCAL/bary/vargrad.jsonl) -ge 125 ] || die "선형 125 스텝 아님"
  python3 $REPO/exe/c1_vargrad_report.py --vg $LOCAL --c1dir $LOCAL/_none --inv /nonexistent \
    --title "ficus 선형·C\$^1\$ none 최적화 변수 기준 기울기 (2026-10-21; C\$^1\$ pl 은 고친 코드로 다시)" \
    --png $RES/ovnet/ficus_c1_vargrad.png --md $RES/ovnet/ficus_c1_vargrad.md >> $L 2>&1 || die "선형 보고 실패"
  mkdir -p $RES/lbours/ficus_c1/vargrad; cp $RES/ovnet/ficus_c1_vargrad.png $RES/ovnet/ficus_c1_vargrad.md $RES/lbours/ficus_c1/vargrad/
  cat >> $RES/outputs.md <<MD

## 2026-10-21 ficus 최적화 변수 θ 기준 기울기 (AF_VARGRAD) -- 선형·C¹ none
- Path: r2:storage/result/anchorflow/lbours/ficus_c1/vargrad/{bary,basenone}/ (vargrad.jsonl, vg_t*.pt: g·θ 벡터·CG 이력, v.log), 그림·표 ~/workspace/result/anchorflow/ovnet/ficus_c1_vargrad.{png,md}
- Commit: eb533ff (AF_VARGRAD), 클러스터 /home/dkta/work/ficus_c1vargrad.sh
- Config: 선형 = lbours/ficus_v0 덤프 역변환 + 다시 풀어 종료 통계 125 스텝, none = c1_base none 첫 스텝 500 회 다시 풀기; CG 100
- 주의: 같은 폴더 c1/ (46 스텝) 은 기울기 버그판 C¹ 측정으로 10-21 지시로 중지 -- 부분 결과, 쓰지 말 것 (클러스터 C1VARGRAD_DONE 표식도 이 부분 결과 기준이라 무효)
MD
  echo C1VARGRAD_BARY_DONE >> $L
fi
# ---- 2) 고친 C¹ 실행 기다림 (최대 48 h)
t0=$(date +%s)
# 10-21 07:15:56 의 C1FIX_FAIL 은 c1fix 감시 오판 (지시): 10-21 07:20:00 이후 시각이 찍힌 줄(과 그 뒤 시각 없는 줄)만 본다.
# 그 구간에서 마지막 표식이 C1FIX_DONE 이면 진행, C1FIX_FAIL 이면 실패 (48 h 상한)
fixstate(){ awk '{ if (match($0, /^[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]/)) ok = (substr($0, 1, 14) >= "10-21 07:20:00") }
                 ok && /C1FIX_FAIL/ {s = "FAIL"} ok && /C1FIX_DONE/ {s = "DONE"} END {print s}' $FIXL 2>/dev/null; }
while true; do
  fs=$(fixstate)
  [ "$fs" = DONE ] && break
  [ "$fs" = FAIL ] && die "07:20 이후 새 C1FIX_FAIL -- 고친 C¹ 측정 안 함"
  [ $(( $(date +%s) - t0 )) -gt 172800 ] && die "C1FIX_DONE 48 h 안 옴"
  sleep 600
done
C=$(grep -o "커밋 [0-9a-f]\{7,\}" $FIXL | tail -1 | awk '{print $2}')
[ -z "$C" ] && C=$($SSH "ls -td /home/dkta/work/wt/c1fix_* 2>/dev/null | head -1" | sed 's#.*/c1fix_##')
[ -z "$C" ] && C=8f602d1
cd $REPO && git fetch -q origin
C=$(git rev-parse --short "$C^{commit}" 2>/dev/null) || die "커밋 $C 없음"
git merge-base --is-ancestor eb533ff $C && git merge-base --is-ancestor 8f602d1 $C || die "커밋 $C 에 AF_VARGRAD 또는 버그 고침이 없음"
log "고친 C¹ 커밋 $C"
$SSH "test -f $RF/N1/roll_ov.pt" || die "덤프 $RF/N1/roll_ov.pt 없음"
if ! $SSH "test -f $RF/vargrad/C1FIXVG_DONE"; then
  $SSH "printf '#!/bin/bash\nC=$C bash /home/dkta/work/ficus_c1vargrad_fix.sh\n' > /home/dkta/work/ficus_c1vargrad_fix_run.sh"
  $SSH "cd /home/dkta/work && AF_DETACH=1 bash anchorflow/exe/job_run.sh /home/dkta/work/ficus_c1vargrad_fix_run.sh /home/dkta/work/ficus_c1vargrad_fix.out" >> $L 2>&1 || die "잡 실행 실패"
  sleep 120
  $SSH "test -s /home/dkta/work/ficus_c1vargrad_fix.out" || die "실행 로그가 안 생김"
  t1=$(date +%s)
  while true; do
    st=$($SSH "if [ -f $RF/vargrad/C1FIXVG_DONE ]; then echo DONE; elif [ -f $RF/vargrad/C1FIXVG_FAIL ]; then echo FAIL; else echo RUN \$(wc -l < $RF/vargrad/c1/vargrad.jsonl 2>/dev/null); fi" 2>/dev/null)
    log "고친 C¹ 측정 $st"
    case "$st" in DONE*) break;; FAIL*) die "고친 C¹ 측정 실패 (클러스터 ficus_c1vargrad_fix.out)";; esac
    [ $(( $(date +%s) - t1 )) -gt 36000 ] && die "고친 C¹ 측정 10 h 초과"
    sleep 600
  done
fi
# ---- 3) 최종 정리
mkdir -p $LOCAL/c1fix
$SSH "cat $RF/vargrad/c1/vargrad.jsonl" > $LOCAL/c1fix/vargrad.jsonl
$SSH "cat $RF/N1/inv_ov.jsonl" > $LOCAL/c1fix/inv_ov.jsonl
python3 $REPO/exe/c1_vargrad_report.py --vg $LOCAL --c1dir $LOCAL/c1fix --inv $LOCAL/c1fix/inv_ov.jsonl --c1name "C¹ pl 고친 코드 (c1fix N1)" \
  --title "ficus C\$^1\$ (기울기 버그 고친 코드, c1fix N1) 최적화 변수 기준 기울기 (2026-10-21, 커밋 $C)" \
  --png $RES/ovnet/ficus_c1_vargrad.png --md $RES/ovnet/ficus_c1_vargrad.md --tex $LOCAL/section.tex >> $L 2>&1 || die "최종 보고 실패"
mkdir -p $RES/lbours/ficus_c1fix/vargrad
cp $RES/ovnet/ficus_c1_vargrad.png $RES/ovnet/ficus_c1_vargrad.md $LOCAL/section.tex $RES/lbours/ficus_c1fix/vargrad/
python3 - $TEX $LOCAL/section.tex <<'PY' >> $L 2>&1
import sys
p, s = sys.argv[1], open(sys.argv[2]).read()
t = open(p).read()
if "c1fix N1) 최적화 변수 기준 기울기" in t:
    print("[tex] 이미 있음 -- 건너뜀")
else:
    i = t.rindex(r"\end{document}")
    open(p, "w").write(t[:i] + s + "\n" + t[i:]); print("[tex] 절 추가", p)
PY
cat >> $RES/outputs.md <<MD

## 2026-10-21 ficus C¹ (기울기 버그 고친 코드) 최적화 변수 θ 기준 기울기 (AF_VARGRAD)
- Path: r2:storage/result/anchorflow/lbours/ficus_c1fix/vargrad/c1/ (vargrad.jsonl, vg_t*.pt, v.log), 그림·표 ~/workspace/result/anchorflow/ovnet/ficus_c1_vargrad.{png,md} (선형·none 겹침)
- Commit: $C, 클러스터 /home/dkta/work/ficus_c1vargrad_fix.sh, 해 = lbours/ficus_c1fix/N1/roll_ov.pt 역변환
- Config: --interp c1wf --c1_barrier 1.0 --c1_jhat 0.3 (c1_base pl), AF_OV_NODE_TGT 로 상태를 덤프에 맞춤, CG 100
- Files: c1_results.tex 절 "ficus C¹ (기울기 버그 고친 코드, c1fix N1) 최적화 변수 기준 기울기"
MD
echo C1VARGRAD_DONE >> $L
