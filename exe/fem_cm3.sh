#!/bin/bash
# 절 CM3 묶음 (커밋 $1, 잡 130567): 결과 불변 가속 (--hspec 주값 분해 30 성분 헤시안·곱, --ncomp 노드 압축) 을 공통 묶음 BA 에 더함 -- 정의 불변.
# 공통 (절 CM2 그대로): HHT ρ 0.8, K 2, 뉴턴 2, εN 1e-2 · εP 1e-1, 묶음 BA (exe/fem_cm3_scn.sh = cm2 BA + --hspec --ncomp).
#  0. 연기 시험 (결정적, 10 프레임, 빈 카드 1-6) -- 하나라도 실패·비유한이면 멈춤
#  V. 결과 같음 (결정적, 빈 카드): cm2 판정 판과 같은 상한 (L000 2x3, ficus 2x20, wolf 2x1, bread 2x1) 에 --hspec / --ncomp / 둘 다 -> cm2 결정적 판 (b·b1) 과 오차 비교
#  A. 단독 시간표 (카드 0, gpu0.lock, cm2 와 같은 규칙: 허용치 0, c 3·6·12·24 -> 맞춤 c*±1 두 번, exe/fem_cm2_sel.py) bread·wolf·L000·ficus  (V 와 동시)
#  B. 바뀐 상한으로 bread·wolf 판정 (결정적, 빈 카드) + 실제 시간 (공통 허용치, 카드 0 단독)
# 결과 R2 lbours/fem_cm/cm3s (연기)·cm3v (같음 확인)·cm3z (시간표)·cm3 (판정)·cm3t (실제 시간), 표 marks/cm3_table.md, 끝 표식 marks/cm3.END
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
C=$1; X=${2:-}; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1                      # X = 다시 돌릴 때 묶음 이름 꼬리 (앞 시도 출력과 섞이지 않게)
PY=/home/dkta/.conda/envs/af/bin/python
B=/dev/shm/dkta/lbours/fem_cm; M=$B/marks; MR=r2:storage/result/anchorflow/lbours/fem_cm/marks; mkdir -p $M /dev/shm/dkta/cm_lists; L=/dev/shm/dkta/cm_lists
LOG=$M/cm3$X.log; say() { echo "$*" | tee -a $LOG; rclone copy $LOG $MR -q; }
say "[cm3 시작] $C $(date '+%F %T')"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> $LOG
export CM_SCN=exe/fem_cm3_scn.sh
source exe/fem_cm3_scn.sh
SC="bread wolf L000 ficus"
for t in $SC; do cm_data $t; done
G="--ga hht --rho 0.8 --K 2 --newton 2"; Z="--tolN 0 --tolP 0"; T="--tolN 1e-2 --tolP 1e-1"
cc0() { case $1 in L000) echo 3;; ficus) echo 20;; *) echo 1;; esac; }                    # cm2 판정 상한
# 0
: > $L/cm3s.txt; for t in $SC; do echo "s $t $G --cg $(cc0 $t) $T --det --frames 10" >> $L/cm3s.txt; done
CM_CARDS="1 2 3 4 5 6" CM_PAR=4 bash exe/fem_cm_pool.sh cm3s$X $L/cm3s.txt > $M/cm3${X}s_pool.log 2>&1
BAD=$($PY - $B/cm3s$X $SC <<'PYE'
import json, os, sys
R = sys.argv[1]; bad = []
for t in sys.argv[2:]:
    p = os.path.join(R, "s_" + t, "summary.json")
    s = json.load(open(p)) if os.path.exists(p) else None
    if s is None or s.get("diverged") or s.get("err_mean") is None: bad.append(t)
print(" ".join(bad))
PYE
)
say "[연기 시험] 실패: '${BAD}' $(date '+%T')"
for t in $SC; do say "  $t: $(grep -a '롤아웃\|rror\|발산' $B/cm3s$X/s_$t/log.txt 2>/dev/null | tail -2 | cut -c1-200 | tr '\n' ' ')"; done
if [ -n "$BAD" ]; then for t in $BAD; do tail -30 $B/cm3s$X/s_$t/log.txt >> $LOG; done; rclone copy $LOG $MR -q
  echo "[cm3 멈춤 -- 연기 시험 실패] $(date '+%F %T')" > $M/cm3$X.END; rclone copy $M/cm3$X.END $MR -q; exit 1; fi
# V (빈 카드) 와 A (카드 0) 동시
: > $L/cm3v.txt
for t in $SC; do c=$(cc0 $t); echo "vh $t $G --cg $c $T --det --hspec" >> $L/cm3v.txt; echo "vn $t $G --cg $c $T --det --ncomp" >> $L/cm3v.txt; echo "vb $t $G --cg $c $T --det --hspec --ncomp" >> $L/cm3v.txt; done
( CM_SCN=exe/fem_cm2_scn.sh CM_CARDS="1 2 3 4 5 6" CM_PAR=6 bash exe/fem_cm_pool.sh cm3v$X $L/cm3v.txt > $M/cm3${X}v_pool.log 2>&1; say "[같음 확인 끝] $(date '+%T')" ) &
PV=$!
: > $L/cm3z1.txt; for c in 3 6 12 24; do for t in $SC; do echo "z${c}_$t $t $G --cg $c $Z" >> $L/cm3z1.txt; done; done
bash exe/fem_cm_q.sh cm3z$X $L/cm3z1.txt > $M/cm3${X}z_q1.log 2>&1; say "[시간표 1] $(date '+%T')"
for r in 2 3; do
  $PY exe/fem_cm2_sel.py next $B/cm3z$X $SC | awk -v g="$G" -v z="$Z" '{print "z"$1"_"$2, $2, g, "--cg", $1, z}' > $L/cm3z$r.txt
  [ -s $L/cm3z$r.txt ] || break
  bash exe/fem_cm_q.sh cm3z$X $L/cm3z$r.txt > $M/cm3z${X}_q$r.log 2>&1; say "[시간표 $r] $(wc -l < $L/cm3z$r.txt) 판 $(date '+%T')"
done
$PY exe/fem_cm2_sel.py pick $B/cm3z$X $SC > $M/cm3${X}_sel.txt 2>&1; cat $M/cm3${X}_sel.txt >> $LOG; rclone copy $M/cm3${X}_sel.txt $MR -q
cap() { awk -v t=$1 '$1==t{print $2}' $M/cm3${X}_sel.txt; }
say "[상한] bread $(cap bread) wolf $(cap wolf) L000 $(cap L000) ficus $(cap ficus)"
# B
: > $L/cm3j.txt; : > $L/cm3t.txt
for t in bread wolf; do c=$(cap $t); [ -z "$c" ] || [ "$c" = 0 ] && { say "[판정] $t 예산 안 상한 없음 -- 판정 안 함"; continue; }
  echo "b $t $G --cg $c $T --det" >> $L/cm3j.txt; echo "a_$t $t $G --cg $c $T" >> $L/cm3t.txt; done
if [ -s $L/cm3j.txt ]; then
  ( CM_CARDS="1 2 3 4 5 6" CM_PAR=6 bash exe/fem_cm_pool.sh cm3$X $L/cm3j.txt > $M/cm3${X}_pool.log 2>&1; say "[판정 끝] $(date '+%T')" ) &
  PJ=$!
  bash exe/fem_cm_q.sh cm3t$X $L/cm3t.txt > $M/cm3${X}t_q.log 2>&1; say "[실제 시간 끝] $(date '+%T')"
  wait $PJ
fi
wait $PV
{ echo "# cm3 ($C) -- 절 CM3 결과 불변 가속 (--hspec --ncomp) 을 공통 묶음에: HHT ρ 0.8, K 2, 뉴턴 2, εN 1e-2 · εP 1e-1, BA = cm2 BA + --hspec --ncomp"; echo
  echo "## 결과 같음 (결정적, cm2 판정 판과 같은 상한: L000 2x3·ficus 2x20·wolf 2x1·bread 2x1; 오차 % / 끝 해시)"
  $PY - $B "$X" $SC <<'PYE'
import json, os, sys
B, X = sys.argv[1], sys.argv[2]
ref = {"L000": ("cm2", "b_L000"), "ficus": ("cm2", "b_ficus"), "wolf": ("cm2", "b1_wolf"), "bread": ("cm2", "b1_bread")}
def one(d):
    p = os.path.join(d, "summary.json")
    if not os.path.exists(p): return "?"
    s = json.load(open(p))
    if s.get("diverged"): return "x%d (%.3f) %s" % (s["diverged"], 100 * s["err_mean_done"], s.get("x_sha1"))
    e = "%.3f %s" % (100 * s["err_mean"], s.get("x_sha1"))
    q = os.path.join(d, "period.txt")
    if os.path.exists(q):
        import re
        m = re.findall(r"정답 대비 ([0-9.]+)\)[^\n]*?진폭 [0-9.]+ m \(비 ([0-9.]+)\)", open(q).read())
        if m: e += " (주기 %s 진폭 %s)" % m[-1]
    return e
print("| 씬 | cm2 (BA) | --hspec | --ncomp | 둘 다 (cm3 BA) |")
print("|---|---|---|---|---|")
for t in sys.argv[3:]:
    r = ref[t]
    print("| %s | %s | %s | %s | %s |" % (t, one(os.path.join(B, r[0], r[1])), one(os.path.join(B, "cm3v" + X, "vh_" + t)), one(os.path.join(B, "cm3v" + X, "vn_" + t)), one(os.path.join(B, "cm3v" + X, "vb_" + t))))
PYE
  echo; echo "## 예산 규칙 시간표 (허용치 0 = 펼친 반복 전부, 카드 0 단독; c 최대 (평균) ms, x = 발산)"; cat $M/cm3${X}_sel.txt; echo
  echo "## 판정 (결정적, 바뀐 상한)"; [ -d $B/cm3$X ] && $PY exe/fem_cm_table.py judge $B/cm3$X
  echo; echo "## 실제 시간 (공통 허용치, 카드 0 단독, 비결정)"; [ -d $B/cm3t$X ] && $PY exe/fem_cm_table.py time $B/cm3t$X
  echo; echo "## 연기 시험 (결정적 10 프레임)"; $PY exe/fem_cm_table.py judge $B/cm3s$X; } > $M/cm3${X}_table.md 2>&1
rclone copy $M $MR -q --include "cm3$X*"
echo "[cm3 끝] $(date '+%F %T')" > $M/cm3$X.END; rclone copy $M/cm3$X.END $MR -q
say "[cm3 끝] $(date '+%F %T')"
