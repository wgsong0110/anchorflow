#!/bin/bash
# 절 CM 묶음 cm1 (커밋 $1): 다섯 형상 공통 설정 -- HHT ρ 0.8, 걸음 K 2, 공통 허용치 εN 1e-2 · εP 1e-1, 반복 상한 하나 (2x20·2x10·2x5·2x3 각각).
#  0. 회귀 (카드 1-6, 결정적): 합친 가지의 재질 판이 implicit-wolf d65370f 와 비트 같은지 (wolf 2.441 % 73fbf2bf7a, bread 0.793 % c6ca35560a)
#  1. 단독 시간 (카드 0, gpu0.lock, 판마다 카드 0 의 다른 프로세스 0): L000·0016·0019·ficus·wolf·bread x 상한 넷 (재질 판은 처치 (a) 켬 = 탄성 판과 같은 HHT, 끔 't' 둘)
#  2. 판정 (카드 1-6, 결정적): 같은 판
#  3. 상한 가운데 L000·0016·0019 < 10 %, 발산 없음, 여섯 씬 단독 최대 프레임 <= 16.7 ms 를 모두 만족하는 가장 큰 상한 -> 나머지 lego 18 (결정적)
# 결과 R2 lbours/fem_cm/cm1 (판정)·cm1t (시간), 표 marks/cm1_table.md, 끝 표식 marks/cm1.END
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
M=/dev/shm/dkta/lbours/fem_cm/marks; MR=r2:storage/result/anchorflow/lbours/fem_cm/marks; mkdir -p $M /dev/shm/dkta/cm_lists
LOG=$M/cm1.log; say() { echo "$*" | tee -a $LOG; rclone copy $LOG $MR -q; }
say "[cm1 시작] $C $(date '+%F %T')"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader >> $LOG
source exe/fem_cm_scn.sh
for t in L000 0016 0019 ficus wolf bread; do cm_data $t; done
T="--tolN 1e-2 --tolP 1e-1"; G="--ga hht --rho 0.8 --K 2"
CAPS="20 10 5 3"
J=/dev/shm/dkta/cm_lists/cm1j.txt; : > $J
echo "reg wolf --ga hht --rho 0.7 --ga_y0ut --K 3 --newton 2 --cg 5 $T --det" >> $J
echo "reg bread --ga hht --rho 0.7 --ga_y0ut --K 3 --newton 2 --cg 5 $T --det" >> $J
for c in $CAPS; do
  for t in L000 0016 0019 ficus wolf bread; do echo "c$c $t $G --newton 2 --cg $c $T --det" >> $J; done
  for t in wolf bread; do echo "c${c}t $t $G --ga_y0ut --newton 2 --cg $c $T --det" >> $J; done
done
( CM_CARDS="1 2 3 4 5 6" CM_PAR=6 bash exe/fem_cm_pool.sh cm1 $J > $M/cm1_pool.log 2>&1; say "[판정 1 끝] $(date '+%F %T')" ) &
PJ=$!
Q=/dev/shm/dkta/cm_lists/cm1t.txt; : > $Q
for c in $CAPS; do
  for t in L000 0016 0019 ficus wolf bread; do echo "c${c}_$t $t $G --newton 2 --cg $c $T" >> $Q; done
  for t in wolf bread; do echo "c${c}t_$t $t $G --ga_y0ut --newton 2 --cg $c $T" >> $Q; done
done
bash exe/fem_cm_q.sh cm1t $Q > $M/cm1t_q.log 2>&1
say "[시간 끝] $(date '+%F %T')"
wait $PJ
SEL=$($PY - /dev/shm/dkta/lbours/fem_cm/cm1 /dev/shm/dkta/lbours/fem_cm/cm1t "$CAPS" <<'PYE'
import json, os, sys
R, RT, caps = sys.argv[1], sys.argv[2], sys.argv[3].split()
def summ(d):
    p = os.path.join(d, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None
def mx(d):
    p = os.path.join(d, "fem.jsonl")
    if not os.path.exists(p): return 1e9
    ms = [json.loads(l).get("ms") for l in open(p)][2:]
    ms = [q for q in ms if q is not None]
    return max(ms) if ms else 1e9
for c in caps:                                                   # 큰 상한부터
    ok = True
    for t in ("L000", "0016", "0019", "ficus", "wolf", "bread"):
        s = summ(os.path.join(R, "c%s_%s" % (c, t)))
        if s is None or s.get("diverged") or s.get("err_mean") is None: ok = False; continue
        if t not in ("ficus", "wolf", "bread") and 100 * s["err_mean"] >= 10: ok = False
        if mx(os.path.join(RT, "c%s_%s" % (c, t))) > 16.7: ok = False
    if ok:
        print(c); break
PYE
)
say "[고른 상한] [$SEL] $(date '+%F %T')"
if [ -n "$SEL" ]; then L2=/dev/shm/dkta/cm_lists/cm1j2.txt; : > $L2
  for t in $(seq -f %04g 0 19); do case $t in 0016|0019) continue;; esac; echo "c$SEL $t $G --newton 2 --cg $SEL $T --det" >> $L2; done
  CM_CARDS="1 2 3 4 5 6" CM_PAR=6 bash exe/fem_cm_pool.sh cm1 $L2 > $M/cm1_pool2.log 2>&1; fi
{ echo "# cm1 ($C) -- 절 CM 다섯 형상 공통 설정 (HHT ρ 0.8, K 2, εN 1e-2 · εP 1e-1, 상한 2x20·2x10·2x5·2x3)"; echo
  echo "## 판정 (결정적, 카드 1-6; 칸 = 평균 오차 %, ficus 는 주기 비·진폭 비 / 오차, x = 발산 프레임; 판 cN = 상한 2xN, cNt = 재질 판 처치 (a) 끔, reg = 재질 판 원래 설정 회귀)"; $PY exe/fem_cm_table.py judge /dev/shm/dkta/lbours/fem_cm/cm1
  echo; echo "회귀 해시: wolf $(grep -o '"x_sha1": "[0-9a-f]*' /dev/shm/dkta/lbours/fem_cm/cm1/reg_wolf/summary.json | cut -c12-21) (implicit-wolf 73fbf2bf7a), bread $(grep -o '"x_sha1": "[0-9a-f]*' /dev/shm/dkta/lbours/fem_cm/cm1/reg_bread/summary.json | cut -c12-21) (c6ca35560a)"
  for d in /dev/shm/dkta/lbours/fem_cm/cm1/*_ficus; do echo "- $(basename $d): $(tail -1 $d/period.txt 2>/dev/null | cut -c1-260)"; done
  echo; echo "## 단독 시간 (카드 0, 비결정, 공통 허용치 실제 전체 프레임; 첫 2 프레임 뺀 평균·최대 ms)"; $PY exe/fem_cm_table.py time /dev/shm/dkta/lbours/fem_cm/cm1t
  echo; echo "고른 상한 (lego 18 추가 판정): [$SEL]"; } > $M/cm1_table.md 2>&1
rclone copy $M $MR -q --include "cm1*"
echo "[cm1 끝] $(date '+%F %T')" > $M/cm1.END; rclone copy $M/cm1.END $MR -q
say "[cm1 끝] $(date '+%F %T')"
