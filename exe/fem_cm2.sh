#!/bin/bash
# 절 CM2 묶음 (커밋 $1, 잡 130567): 정의 변경 셋 (사용자 승인 11-05) -- (1) 반복 상한 = 프레임 예산 16.7 ms 규칙, (2) 재질 판도 같은 실행기·같은 인자 (장벽 바닥·점유 가중·MG·--mgbar 2·--mv2·--lsfast·--fk), (3) 뉴턴 시작점 = 관성 예측 (--ga_y0ut).
# 공통: HHT ρ 0.8, K 2, 뉴턴 2, εN 1e-2 · εP 1e-1, 묶음 BA (exe/fem_cm2_scn.sh). 씬별 PCG 상한 c = 허용치 0 (펼친 반복 전부) 단독 전체 궤적 최대 프레임 <= 16.7 ms 인 가장 큰 c (lego 궤적은 L000·0016·0019 중 작은 c 하나).
#  A. 단독 시간표 (카드 0, gpu0.lock, 판마다 카드 0 의 다른 계산 프로세스 0): c 3·6·12·24 -> 선형 맞춤 예측 c* 와 ±1 (두 번) -> 씬별 c
#  B. 판정 (결정적, 카드 1-6) L000·0016·0019·drop1·ficus·wolf·bread + 같은 판 실제 시간 (공통 허용치, 카드 0)
#  C. 기준 (lego 셋 < 10 %, ficus 주기 비 0.96..1.04, wolf·bread 발산 없음 < 10 %, 실제 최대 <= 16.7) 을 모두 만족하면 lego 나머지 18 판정 + 허용치 0 시간
# 결과 R2 lbours/fem_cm/cm2z (시간표)·cm2 (판정)·cm2t (실제 시간)·cm2l (lego 18 시간), 표 marks/cm2_table.md, 끝 표식 marks/cm2.END
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8 CM_SCN=exe/fem_cm2_scn.sh
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
B=/dev/shm/dkta/lbours/fem_cm; M=$B/marks; MR=r2:storage/result/anchorflow/lbours/fem_cm/marks; mkdir -p $M /dev/shm/dkta/cm_lists; L=/dev/shm/dkta/cm_lists
LOG=$M/cm2.log; say() { echo "$*" | tee -a $LOG; rclone copy $LOG $MR -q; }
say "[cm2 시작] $C $(date '+%F %T')"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> $LOG
source exe/fem_cm2_scn.sh
SC="L000 0016 0019 drop1 ficus wolf bread"
for t in $SC; do cm_data $t; done
G="--ga hht --rho 0.8 --K 2 --newton 2"; Z="--tolN 0 --tolP 0"; T="--tolN 1e-2 --tolP 1e-1"
# A
: > $L/cm2z1.txt; for c in 3 6 12 24; do for t in $SC; do echo "z${c}_$t $t $G --cg $c $Z" >> $L/cm2z1.txt; done; done
bash exe/fem_cm_q.sh cm2z $L/cm2z1.txt > $M/cm2z_q1.log 2>&1; say "[시간표 1] $(date '+%T')"
for r in 2 3; do
  $PY exe/fem_cm2_sel.py next $B/cm2z $SC | awk -v g="$G" -v z="$Z" '{print "z"$1"_"$2, $2, g, "--cg", $1, z}' > $L/cm2z$r.txt
  [ -s $L/cm2z$r.txt ] || break
  bash exe/fem_cm_q.sh cm2z $L/cm2z$r.txt > $M/cm2z_q$r.log 2>&1; say "[시간표 $r] $(wc -l < $L/cm2z$r.txt) 판 $(date '+%T')"
done
$PY exe/fem_cm2_sel.py pick $B/cm2z $SC > $M/cm2_sel.txt 2>&1; cat $M/cm2_sel.txt >> $LOG; rclone copy $M/cm2_sel.txt $MR -q
cap() { awk -v t=$1 '$1==t{print $2}' $M/cm2_sel.txt; }
CL=$(for t in L000 0016 0019; do cap $t; done | sort -n | head -1)
say "[상한] lego $CL drop1 $(cap drop1) ficus $(cap ficus) wolf $(cap wolf) bread $(cap bread)"
cc() { case $1 in drop1|ficus|wolf|bread) cap $1;; *) echo $CL;; esac; }
# B
: > $L/cm2j.txt; : > $L/cm2t.txt
for t in $SC; do c=$(cc $t); [ -z "$c" ] || [ "$c" = 0 ] && continue
  echo "b $t $G --cg $c $T --det" >> $L/cm2j.txt; echo "a_$t $t $G --cg $c $T" >> $L/cm2t.txt; done
( CM_CARDS="1 2 3 4 5 6" CM_PAR=6 bash exe/fem_cm_pool.sh cm2 $L/cm2j.txt > $M/cm2_pool.log 2>&1; say "[판정 끝] $(date '+%T')" ) &
PJ=$!
bash exe/fem_cm_q.sh cm2t $L/cm2t.txt > $M/cm2t_q.log 2>&1; say "[실제 시간 끝] $(date '+%T')"
wait $PJ
# C
OK=$($PY - $B/cm2 $B/cm2t <<'PYE'
import json, os, re, sys
R, RT = sys.argv[1], sys.argv[2]
ok = True
for t in ("L000", "0016", "0019", "drop1", "ficus", "wolf", "bread"):
    p = os.path.join(R, "b_" + t, "summary.json")
    s = json.load(open(p)) if os.path.exists(p) else None
    if s is None or s.get("diverged") or s.get("err_mean") is None: ok = False; continue
    if t in ("L000", "0016", "0019", "wolf", "bread") and 100 * s["err_mean"] >= 10: ok = False
    if t == "ficus":
        m = re.findall(r"정답 대비 ([0-9.]+)\)", open(os.path.join(R, "b_ficus", "period.txt")).read()) if os.path.exists(os.path.join(R, "b_ficus", "period.txt")) else []
        if not m or not (0.96 <= float(m[-1]) <= 1.04): ok = False
    p = os.path.join(RT, "a_" + t, "fem.jsonl")
    ms = [json.loads(l).get("ms") for l in open(p)][2:] if os.path.exists(p) else []
    ms = [q for q in ms if q is not None]
    if not ms or max(ms) > 16.7: ok = False
print(1 if ok else 0)
PYE
)
say "[기준] $OK $(date '+%T')"
if [ "$OK" = 1 ]; then
  : > $L/cm2l.txt; : > $L/cm2lz.txt
  for t in $(seq -f %04g 0 19); do case $t in 0016|0019) continue;; esac; echo "b $t $G --cg $CL $T --det" >> $L/cm2l.txt; echo "z${CL}_$t $t $G --cg $CL $Z" >> $L/cm2lz.txt; done
  ( CM_CARDS="1 2 3 4 5 6" CM_PAR=6 bash exe/fem_cm_pool.sh cm2 $L/cm2l.txt > $M/cm2_pool2.log 2>&1; say "[lego 18 판정 끝] $(date '+%T')" ) &
  PJ=$!
  bash exe/fem_cm_q.sh cm2l $L/cm2lz.txt > $M/cm2l_q.log 2>&1; say "[lego 18 시간 끝] $(date '+%T')"
  wait $PJ
fi
{ echo "# cm2 ($C) -- 절 CM2 정의 변경 셋 (예산 상한·재질 판 같은 실행기·시작점 관성 예측): HHT ρ 0.8, K 2, 뉴턴 2, εN 1e-2 · εP 1e-1, 묶음 BA"; echo
  echo "## 예산 규칙 시간표 (허용치 0 = 펼친 반복 전부, 카드 0 단독; c 최대 (평균) ms, x = 발산)"; cat $M/cm2_sel.txt; echo "lego 상한 = $CL"; echo
  echo "## 판정 (결정적, 카드 1-6)"; $PY exe/fem_cm_table.py judge $B/cm2
  for d in $B/cm2/b_ficus; do echo "- ficus: $(tail -1 $d/period.txt 2>/dev/null | cut -c1-260)"; done
  for d in $B/cm2/b_drop1; do echo "- drop1: $(tail -1 $d/drop_ana.txt 2>/dev/null | cut -c1-260)"; done
  echo; echo "## 실제 시간 (공통 허용치, 카드 0 단독, 비결정)"; $PY exe/fem_cm_table.py time $B/cm2t
  [ -d $B/cm2l ] && { echo; echo "## lego 18 허용치 0 시간 (카드 0 단독)"; $PY exe/fem_cm_table.py time $B/cm2l; }
  echo; echo "기준 통과 $OK"; } > $M/cm2_table.md 2>&1
rclone copy $M $MR -q --include "cm2*"
echo "[cm2 끝] $(date '+%F %T')" > $M/cm2.END; rclone copy $M/cm2.END $MR -q
say "[cm2 끝] $(date '+%F %T')"
