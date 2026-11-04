#!/bin/bash
# 절 AN.22 과제 e1i (커밋 $1): 최종 공통 설정 (HHT ρ0.8 + 장벽 띠 고착 + κ 0.1 + --mv2 --lsfast --mgbar 2 --fk, 공통 허용치 εN 1e-2 · εP 1e-1, 시간 상한 lego·낙하 2x3, ficus 2x20)
#  1. 판정 (카드 1-6, 결정적): L000·0016·0019·drop1 -> 모두 < 10 % 이고 발산 없으면 나머지 lego 18 + ficus 125 프레임
#  2. 단독 시간 (카드 0): 최악 (허용치 0) L000 10 프레임 2 회, 실제 (공통 허용치) lego 21 궤적 80 프레임 + drop1 + ficus 125 프레임
# 결과 R2 lbours/fem_an/e1/ (판정), e1t/ (시간), 표 marks/e1_table.md, 끝 표식 marks/e1.END
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
C=$1; D=/dev/shm/dkta/impwt/$C; cd $D || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_e1i XDG_CACHE_HOME=/dev/shm/dkta/cache; mkdir -p $CUPY_CACHE_DIR /dev/shm/dkta/an_lists
M=/dev/shm/dkta/lbours/fem_an/marks; MR=r2:storage/result/anchorflow/lbours/fem_an/marks; mkdir -p $M
LOG=$M/e1.log; say() { echo "$*" | tee -a $LOG; rclone copy $LOG $MR -q; }
say "[e1 시작] $C $(date '+%F %T')"
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mg_lreuse --mv2 --lsfast --fk"
T="--tolN 1e-2 --tolP 1e-1"; T0="--tolN 0 --tolP 0"
H8="--ga hht --rho 0.8 --fl_bar --fl_bst 1 --gq_oct 0.1 --mgbar 2"
FIA="--ga hht --rho 0.8 --gq_oct 0.1 --mgbar 2 --newton 2 --cg 20"
L=/dev/shm/dkta/an_lists/e1a1.txt; : > $L
for t in L000 0016 0019 drop1; do echo "k23 $t $B --det $H8 $T --newton 2 --cg 3" >> $L; done
echo "fik ficus $B --det $FIA $T" >> $L
( AN_CARDS="1 2 3 4 5 6" AN_PAR=6 AN_GAP=4 bash exe/fem_an_pool.sh e1 $L > $M/e1a_pool.log 2>&1
  SEL=$($PY - /dev/shm/dkta/lbours/fem_an/e1 <<'PYE'
import json, os, sys
R = sys.argv[1]; ok = True
for t in ("L000", "0016", "0019", "drop1"):
    p = os.path.join(R, "k23_" + t, "summary.json")
    if not os.path.exists(p): ok = False; continue
    d = json.load(open(p))
    if d.get("diverged") or d.get("div_at") or (t != "drop1" and 100 * d["err_mean"] >= 10): ok = False
print("k23" if ok else "")
PYE
)
  say "[판정 1 단계 끝] 고른 판 [$SEL] $(date '+%F %T')"
  if [ -n "$SEL" ]; then L2=/dev/shm/dkta/an_lists/e1a2.txt; : > $L2
    for t in $(seq -f %04g 0 19); do case $t in 0016|0019) continue;; esac; echo "k23 $t $B --det $H8 $T --newton 2 --cg 3" >> $L2; done
    AN_CARDS="1 2 3 4 5 6" AN_PAR=6 AN_GAP=4 bash exe/fem_an_pool.sh e1 $L2 > $M/e1b_pool.log 2>&1; fi
  say "[판정 끝] $(date '+%F %T')" ) &
PJ=$!
Q=/dev/shm/dkta/an_lists/e1t.txt; : > $Q
for r in 1 2; do echo "lw_k23_r$r @LG --frames 10 $T0 $H8 --newton 2 --cg 3" >> $Q; done
echo "dw_k23 @DR --frames 10 $T0 $H8 --newton 2 --cg 3" >> $Q
echo "fw_k220 @FI --frames 20 $T0 $FIA" >> $Q
echo "fr_k220 @FI --frames 125 $T $FIA" >> $Q
echo "dr_k23 @DR --frames 80 $T $H8 --newton 2 --cg 3" >> $Q
echo "lr_k23_L000 @LG --frames 80 $T $H8 --newton 2 --cg 3" >> $Q
LD=/tmp/dkta/af_data; [ -f $LD/0019.pt ] || LD=/dev/shm/dkta/anerr_data
for t in $(seq -f %04g 0 19); do [ -s $LD/$t.pt ] || cp /home/dkta/work/lbpt/test/$t.pt $LD/$t.pt; echo "lr_k23_$t --data $LD/$t.pt --floor vel --tag $t --frames 80 $T $H8 --newton 2 --cg 3" >> $Q; done
echo "pf_k23 @LG --frames 6 --prof 4 $T0 $H8 --newton 2 --cg 3" >> $Q
bash exe/fem_an_e1q.sh e1t $Q > $M/e1t_q.log 2>&1
cp /dev/shm/dkta/lbours/fem_an/e1t/run.out $M/e1_time.md; rclone copy $M/e1_time.md $MR -q
say "[시간 끝] $(date '+%F %T')"
wait $PJ
{ echo "# e1 ($C) -- 절 AN.22 최종 공통 설정 판정·시간"; echo
  echo "## 판정 (결정적; lego·낙하 상한 2x3, ficus 2x20)"; $PY exe/fem_an_tol_table.py /dev/shm/dkta/lbours/fem_an/e1
  for d in /dev/shm/dkta/lbours/fem_an/e1/*_drop1; do v=$(basename $d); echo "- ${v%_drop1} 낙하: $(tail -1 $d/drop_ana.txt 2>/dev/null | cut -c1-220)"; done
  for d in /dev/shm/dkta/lbours/fem_an/e1/*_ficus; do echo "- ficus: $(tail -2 $d/period.txt 2>/dev/null | tr '\n' ' ' | cut -c1-300)"; done
  echo; echo "## 단독 시간 (카드 0, 비결정; lw = 최악 허용치 0 10 프레임, lr·dr·fr = 실제 공통 허용치 전체 프레임, 첫 2 프레임 뺀 평균·최대)"; cat $M/e1_time.md; } > $M/e1_table.md 2>&1
rclone copy $M $MR -q --include "e1*"
echo "[e1 끝] $(date '+%F %T')" > $M/e1.END; rclone copy $M/e1.END $MR -q
say "[e1 끝] $(date '+%F %T')"
