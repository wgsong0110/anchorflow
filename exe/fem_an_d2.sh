#!/bin/bash
# 절 AN.21 묶음 d2 (잡 안; d1 결과 뒤): HHT ρ 0.8 + 장벽 + 띠 고착 (s1) 이 수렴 예산 (8x40) 에서 L000 3.69·0016 10.07 %·반발 0.708 인데 1x5 에서 발산 -> 뉴턴 x PCG 상한 의존을 가름.
#  1. 단독 시간 (카드 0): HHT s1 덩어리·비덩어리 1x7 최악 lego L000 10 프레임 2 회, drop1 1x7 최악
#  2. 판정 (카드 1 에서 6, 결정적, d1 과 같은 공통 인자): h8 s1 l·f x 상한 1x6·1x7·2x5·2x10·4x20 x drop1·0016·0019·L000 + ficus h8 (2x20, κ 0.1)
#  3. 16.7 ms 안 상한 (1x6·1x7) 중 L000·0016·0019 모두 < 10 % 이고 발산 없는 판 (평균 작은 순 둘) -> 나머지 lego 18
# 결과 R2 lbours/fem_an/d2/, d2t/, 표 marks/d2_table.md, 끝 표식 marks/d2.END
cd "$(dirname "$0")/.." || exit 1
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_d2 XDG_CACHE_HOME=/dev/shm/dkta/cache
PY=/home/dkta/.conda/envs/af/bin/python
M=/dev/shm/dkta/lbours/fem_an/marks; mkdir -p $M /dev/shm/dkta/an_lists $CUPY_CACHE_DIR
MR=r2:storage/result/anchorflow/lbours/fem_an/marks
SD=/dev/shm/dkta/anerr_data; FD=$SD/F_pg_v0.pt
R=/dev/shm/dkta/lbours/fem_an/d2; RT=/dev/shm/dkta/lbours/fem_an/d2t; mkdir -p $R $RT
RRT=r2:storage/result/anchorflow/lbours/fem_an/d2t
LOG=$M/d2.log
say() { echo "$*" | tee -a $LOG; rclone copy $LOG $MR -q; }
say "[d2 시작] $(cat .done) $(date '+%F %T')"
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mg_lreuse --mv2"
T="--tolN 1e-2 --tolP 1e-1"; T0="--tolN 0 --tolP 0"; TC="--tolN 1e-6 --tolP 1e-3"
OK="--gq_oct 0.1"
LG="--data $SD/L000.pt --floor vel --tag L000"
U0=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
others0() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$U0"; }
trun() { local v=$1; shift; local O=$RT/$v; mkdir -p $O
  while [ "$(others0)" -gt 0 ]; do sleep 5; done
  echo "시작 때 카드 0 의 다른 프로세스 $(others0) 개 $(date +%T)" > $O/card.txt
  CUDA_VISIBLE_DEVICES=0 timeout 1800 $PY -u exe/fem_an_rt_sim.py --out $O $B "$@" 9>&- < /dev/null > $O/log.txt 2>&1
  echo "끝 때 카드 0 의 다른 프로세스 $(others0) 개 $(date +%T)" >> $O/card.txt
  echo "$v | $(grep -a '롤아웃\|rror\|발산' $O/log.txt | tail -2 | cut -c1-160 | tr '\n' ' ') | $($PY -c "import json;d=json.load(open('$O/summary.json'));print('sha',d['x_sha1'],'mean',round(d.get('ms_mean') or -1,2),'max',round(d.get('ms_max') or -1,2),'p90',round(d.get('ms_p90') or -1,2),'nN',(d.get('it_nN') or {}).get('mean'),'nP',(d.get('it_nP') or {}).get('mean'),'nPmax',(d.get('it_nP') or {}).get('max'))" 2>/dev/null) | $(cat $O/card.txt | tr '\n' ' ')" >> $RT/run.out
  grep -q Error $O/log.txt && tail -15 $O/log.txt >> $RT/run.out
  rclone copy $O $RRT/$v -q; rclone copy $RT/run.out $RRT -q; }

exec 9>/home/dkta/work/gpu0.lock; flock 9
L=/dev/shm/dkta/an_lists/d2a.txt; : > $L
H8="--ga hht --rho 0.8 --fl_bar --fl_bst 1"
for hh in "l:" "f:--bar_full"; do hn=${hh%%:*}; H=${hh#*:}
  for cap in 1x7 1x6 2x5 2x10 4x20; do n=${cap%x*}; c=${cap#*x}
    for t in drop1 0016 0019 L000; do echo "h8s1${hn}c$cap $t $B --det $H8 $H $OK $T --newton $n --cg $c" >> $L; done; done; done
echo "fih8 ficus $B --det --ga hht --rho 0.8 $OK $T --newton 2 --cg 20" >> $L
say "판정 목록 $(wc -l < $L) 줄"
( AN_CARDS="1 2 3 4 5 6" AN_PAR=6 AN_GAP=5 bash exe/fem_an_pool.sh d2 $L > $M/d2a_pool.log 2>&1
  $PY exe/fem_an_tol_table.py $R > $M/d2a_table.md 2>&1
  rclone copy $M $MR -q --include "d2*" ) 9>&- &
PJ=$!
DR="--data $SD/ak_drop2/drop_f1.pt --floor vel --tag drop1"; [ -s /tmp/dkta/ak_drop2/drop_f1.pt ] && DR="--data /tmp/dkta/ak_drop2/drop_f1.pt --floor vel --tag drop1"
for rep in 1 2; do
  trun lwc_h8_s1l_1x7_r$rep $LG --frames 10 $T0 --newton 1 --cg 7 $OK $H8
  trun lwc_h8_s1f_1x7_r$rep $LG --frames 10 $T0 --newton 1 --cg 7 $OK $H8 --bar_full
  trun dwc_h8_s1f_1x7_r$rep $DR --frames 10 $T0 --newton 1 --cg 7 $OK $H8 --bar_full
done
for hh in "l:" "f:--bar_full"; do hn=${hh%%:*}; H=${hh#*:}; trun lr_h8_s1${hn}_1x7 $LG --frames 80 $T --newton 1 --cg 7 $OK $H8 $H; done
flock -u 9; exec 9>&-
cp $RT/run.out $M/d2_time.md; rclone copy $M/d2_time.md $MR -q
say "[시간 끝] $(date '+%F %T')"
wait $PJ
say "[판정 1 단계 끝] $(date '+%F %T')"
# ---- 3. 16.7 ms 안 상한의 좋은 판 -> lego 21 전체 ----
SEL=$($PY - $R <<'PYE'
import json, os, sys
R = sys.argv[1]; vs = {}
for d in os.listdir(R):
    if "_" not in d: continue
    v, t = d.rsplit("_", 1)
    p = os.path.join(R, d, "summary.json")
    if os.path.exists(p): vs.setdefault(v, {})[t] = json.load(open(p))
ok = []
for v, r in vs.items():
    if not (v.endswith("c1x6") or v.endswith("c1x7")) or not all(t in r for t in ("L000", "0016", "0019", "drop1")): continue
    if any(r[t].get("diverged") for t in r): continue
    e = [100 * r[t]["err_mean"] for t in ("L000", "0016", "0019")]
    if max(e) < 10: ok.append((sum(e) / 3, v))
ok.sort(); print(" ".join(v for _, v in ok[:2]))
PYE
)
say "3 단계 고른 판 (16.7 ms 안 상한, L000·0016·0019 < 10 %, 발산 없음): [$SEL]"
if [ -n "$SEL" ]; then
  L=/dev/shm/dkta/an_lists/d2b.txt; : > $L
  for v in $SEL; do A=$(grep -m1 "^$v 0016 " /dev/shm/dkta/an_lists/d2a.txt | cut -d' ' -f3-)
    for t in $(seq -f %04g 0 19); do case $t in 0016|0019) continue;; esac; echo "$v $t $A" >> $L; done; done
  AN_CARDS="1 2 3 4 5 6" AN_PAR=6 AN_GAP=5 bash exe/fem_an_pool.sh d2 $L > $M/d2b_pool.log 2>&1
fi
{ echo "# d2 ($(cat .done)) -- 절 AN.21 HHT ρ 0.8 + 장벽 + 띠 고착 상한 의존; 3 단계 고른 판 [$SEL]"; echo
  echo "## 판정 (결정적, --mv2 --gq_oct 0.1, εN 1e-2 · εP 1e-1; 이름 h8s1 + l 덩어리·f 비덩어리 + c 상한)"
  $PY exe/fem_an_tol_table.py $R
  for d in $R/*_drop1; do v=$(basename $d); echo "- ${v%_drop1} 낙하: $(tail -1 $d/drop_ana.txt 2>/dev/null | cut -c1-200)"; done
  echo; echo "## 단독 시간 (카드 0, 비결정)"; cat $RT/run.out; } > $M/d2_table.md 2>&1
say "[d2 끝] $(date '+%F %T')"
echo "[d2 끝] $(date '+%F %T') 고른 판 [$SEL]" > $M/d2.END
rclone copy $M $MR -q --include "d2*"
