#!/bin/bash
# 절 CM2 묶음 cm2b (커밋 $1): 예산 규칙에서 상한이 없거나 작은 재질 씬 -- wolf·bread 를 c 1·3 으로 판정 (결정적, 카드 1-6) + 실제 시간 (카드 0 단독),
# 그리고 커널별 시간 (--prof, 카드 0 단독, 허용치 0 c 3; 그래프 밖 한 번이라 절대값은 재생보다 큼 -- 비율만) L000·wolf·bread.
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8 CM_SCN=exe/fem_cm2_scn.sh
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
B=/dev/shm/dkta/lbours/fem_cm; M=$B/marks; MR=r2:storage/result/anchorflow/lbours/fem_cm/marks; L=/dev/shm/dkta/cm_lists; mkdir -p $M $L
LOG=$M/cm2b.log; say() { echo "$*" | tee -a $LOG; rclone copy $LOG $MR -q; }
say "[cm2b 시작] $C $(date '+%F %T')"
G="--ga hht --rho 0.8 --K 2 --newton 2"; Z="--tolN 0 --tolP 0"; T="--tolN 1e-2 --tolP 1e-1"
: > $L/cm2bj.txt; : > $L/cm2bt.txt
for c in 1 3; do for t in wolf bread; do echo "b$c $t $G --cg $c $T --det" >> $L/cm2bj.txt; echo "a${c}_$t $t $G --cg $c $T" >> $L/cm2bt.txt; done; done
for t in L000 wolf bread; do echo "p3_$t $t $G --cg 3 $Z --prof 10" >> $L/cm2bt.txt; done
( CM_CARDS="1 2 3 4 5 6" CM_PAR=6 bash exe/fem_cm_pool.sh cm2 $L/cm2bj.txt > $M/cm2b_pool.log 2>&1; say "[판정 끝] $(date '+%T')" ) &
PJ=$!
bash exe/fem_cm_q.sh cm2bt $L/cm2bt.txt > $M/cm2bt_q.log 2>&1; say "[시간 끝] $(date '+%T')"
wait $PJ
{ echo "# cm2b ($C) -- wolf·bread c 1·3 판정·실제 시간, 커널별 시간"; $PY exe/fem_cm_table.py judge $B/cm2; echo; $PY exe/fem_cm_table.py time $B/cm2bt
  for t in L000 wolf bread; do echo; echo "## 커널별 (p3_$t, 프레임 평균 ms, 상위 15)"; $PY - $B/cm2bt/p3_$t/fem.jsonl <<'PYE'
import json, sys, collections
acc = collections.Counter(); n = 0
for l in open(sys.argv[1]):
    d = json.loads(l)
    k = d.get("kernel_ms")
    if not k: continue
    n += 1
    for a, b in k.items():
        acc[a] += (b if isinstance(b, (int, float)) else (b[0] if isinstance(b, list) else 0))
tot = sum(acc.values())
print("프레임", n, "합 %.2f ms" % (tot / max(n, 1)))
for a, b in acc.most_common(15): print("- %s %.3f ms (%.0f %%)" % (a, b / max(n, 1), 100 * b / max(tot, 1e-9)))
PYE
  done; } > $M/cm2b_table.md 2>&1
rclone copy $M $MR -q --include "cm2b*"
echo "[cm2b 끝] $(date '+%F %T')" > $M/cm2b.END; rclone copy $M/cm2b.END $MR -q; say "[cm2b 끝] $(date '+%F %T')"
