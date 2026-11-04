#!/bin/bash
# 절 AN.21 묶음 d1 (잡 안): 장벽 바닥 (--fl_bar) 위 띠 고착 (--fl_bst, 정의 변경) + 비덩어리 장벽 헤시안 (--bar_full).
#  0. 스모크 (카드 0, 결정적, L000 4 프레임): 새 플래그 컴파일·완주 -- 실패면 끝냄 (marks/d1_smoke.md)
#  1. 단독 시간 (카드 0, gpu0.lock, 판마다 카드 0 의 다른 계산 프로세스 0 확인, 비결정, --mv2 --gq_oct 0.1): lego L000 최악 (허용치 0) 10 프레임 2 회 + 상한 훑기, 실제 (공통 허용치) 80 프레임
#  2. 판정 (카드 1 에서 6, 결정적, --mv2 --gq_oct 0.1, 공통 허용치 εN 1e-2 · εP 1e-1, lego·낙하 상한 1 x 5): drop1·L000·0016·0019 x 적분기 (후퇴 K 2, HHT ρ 0.7·0.8) x 바닥 (bst 1·2·1+점별) x 헤시안 (덩어리·비덩어리)
#     + 대조 (κ 없음), 수렴 예산 (8 x 40, 1e-6 / 1e-3)
#  3. 2 에서 L000·0016·0019 모두 < 10 % 이고 발산 없는 판 (세 평균 작은 순 셋) -> 나머지 lego 18 + ficus (그 적분기, 2 x 20, --floor none 이라 바닥 플래그 무관)
# 결과 R2 lbours/fem_an/d1/, d1t/ (시간), 표 marks/d1_table.md, 끝 표식 marks/d1.END
cd "$(dirname "$0")/.." || exit 1
export PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH PYTHONUTF8=1 OMP_NUM_THREADS=8
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_d1 XDG_CACHE_HOME=/dev/shm/dkta/cache
PY=/home/dkta/.conda/envs/af/bin/python
M=/dev/shm/dkta/lbours/fem_an/marks; mkdir -p $M /dev/shm/dkta/an_lists $CUPY_CACHE_DIR
MR=r2:storage/result/anchorflow/lbours/fem_an/marks
SD=/dev/shm/dkta/anerr_data; FD=$SD/F_pg_v0.pt
R=/dev/shm/dkta/lbours/fem_an/d1; RT=/dev/shm/dkta/lbours/fem_an/d1t; mkdir -p $R $RT
RRT=r2:storage/result/anchorflow/lbours/fem_an/d1t
LOG=$M/d1.log
say() { echo "$*" | tee -a $LOG; rclone copy $LOG $MR -q; }
say "[d1 시작] $(cat .done) $(date '+%F %T')"
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

# ---- 0. 스모크 ----
exec 9>/home/dkta/work/gpu0.lock; flock 9
for sv in "s_b1:--integ be --fl_bar --fl_bst 1" "s_b2f:--integ be --fl_bar --fl_bst 2 --bar_full" "s_h1f:--ga hht --rho 0.8 --fl_bar --fl_bst 1 --bar_full --fl_pt"; do
  v=${sv%%:*}; I=${sv#*:}
  trun ${v}_det $LG --frames 4 $T --newton 1 --cg 5 $OK $I --det
  trun ${v}_nd $LG --frames 4 $T --newton 1 --cg 5 $OK $I
done
cp $RT/run.out $M/d1_smoke.md; rclone copy $M/d1_smoke.md $MR -q
if grep -q "rror\|발산" $M/d1_smoke.md || [ "$(grep -c 'sha' $M/d1_smoke.md)" -lt 6 ]; then
  say "[d1 스모크 실패] $(date '+%F %T')"; echo "스모크 실패" > $M/d1.END; rclone copy $M/d1.END $MR -q; exit 1; fi
say "[스모크 통과] $(date '+%F %T')"

# ---- 2. 판정 (카드 1 에서 6, 시간 측정과 동시; 풀은 카드 0 을 쓰지 않음) ----
L=/dev/shm/dkta/an_lists/d1a.txt; : > $L
IS=("be:--integ be" "h8:--ga hht --rho 0.8" "h7:--ga hht --rho 0.7")
FS=("s1:--fl_bar --fl_bst 1" "s2:--fl_bar --fl_bst 2" "s1p:--fl_bar --fl_bst 1 --fl_pt")
HS=("l:" "f:--bar_full")
for ii in "${IS[@]}"; do i=${ii%%:*}; I=${ii#*:}
  for ff in "${FS[@]}"; do fn=${ff%%:*}; F=${ff#*:}
    for hh in "${HS[@]}"; do hn=${hh%%:*}; H=${hh#*:}
      for t in drop1 0016 0019 L000; do echo "$i$fn$hn $t $B --det $I $F $H $OK $T --newton 1 --cg 5" >> $L; done; done; done; done
for ii in "${IS[@]:0:2}"; do i=${ii%%:*}; I=${ii#*:}                                  # 대조: κ 없음 (점유 가중 끔)
  for t in drop1 0016 0019 L000; do echo "${i}s1fno $t $B --det $I --fl_bar --fl_bst 1 --bar_full $T --newton 1 --cg 5" >> $L; done
  for t in 0016 L000 drop1; do echo "${i}s1fcv $t $B --det $I --fl_bar --fl_bst 1 --bar_full $OK $TC --newton 8 --cg 40" >> $L; done; done   # 수렴 예산
say "판정 목록 $(wc -l < $L) 줄"
( AN_CARDS="1 2 3 4 5 6" AN_PAR=6 AN_GAP=5 bash exe/fem_an_pool.sh d1 $L > $M/d1a_pool.log 2>&1
  $PY exe/fem_an_tol_table.py $R > $M/d1a_table.md 2>&1
  for d in $R/*_drop1; do v=$(basename $d); echo "## ${v%_drop1} 낙하 $(tail -1 $d/drop_ana.txt 2>/dev/null | cut -c1-200)"; done >> $M/d1a_table.md
  rclone copy $M $MR -q --include "d1*" ) 9>&- &
PJ=$!

# ---- 1. 단독 시간 (카드 0) ----
for rep in 1 2; do
  trun lw_be_bar_r$rep $LG --frames 10 $T0 --newton 1 --cg 5 $OK --integ be
  trun lw_be_s1l_r$rep $LG --frames 10 $T0 --newton 1 --cg 5 $OK --integ be --fl_bar --fl_bst 1
  trun lw_be_s1f_r$rep $LG --frames 10 $T0 --newton 1 --cg 5 $OK --integ be --fl_bar --fl_bst 1 --bar_full
  trun lw_h8_s1l_r$rep $LG --frames 10 $T0 --newton 1 --cg 5 $OK --ga hht --rho 0.8 --fl_bar --fl_bst 1
  trun lw_h8_s1f_r$rep $LG --frames 10 $T0 --newton 1 --cg 5 $OK --ga hht --rho 0.8 --fl_bar --fl_bst 1 --bar_full
  trun lw_h8_nob_r$rep $LG --frames 10 $T0 --newton 1 --cg 5 $OK --ga hht --rho 0.8
done
for cap in 1x6 1x7 2x3 2x4 2x5; do n=${cap%x*}; c=${cap#*x}
  trun lwc_h8_s1f_$cap $LG --frames 10 $T0 --newton $n --cg $c $OK --ga hht --rho 0.8 --fl_bar --fl_bst 1 --bar_full
done
for sv in "be_s1l:--integ be --fl_bar --fl_bst 1" "be_s1f:--integ be --fl_bar --fl_bst 1 --bar_full" "h8_s1l:--ga hht --rho 0.8 --fl_bar --fl_bst 1" "h8_s1f:--ga hht --rho 0.8 --fl_bar --fl_bst 1 --bar_full"; do
  v=${sv%%:*}; I=${sv#*:}; trun lr_$v $LG --frames 80 $T --newton 1 --cg 5 $OK $I; done
flock -u 9; exec 9>&-
cp $RT/run.out $M/d1_time.md; rclone copy $M/d1_time.md $MR -q
say "[시간 끝] $(date '+%F %T')"
wait $PJ
say "[판정 1 단계 끝] $(date '+%F %T')"

# ---- 3. 좋은 판 -> lego 21 전체 + ficus ----
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
    if v.endswith("cv") or not all(t in r for t in ("L000", "0016", "0019", "drop1")): continue
    if any(r[t].get("diverged") for t in r): continue
    e = [100 * r[t]["err_mean"] for t in ("L000", "0016", "0019")]
    if max(e) < 10: ok.append((sum(e) / 3, v))
ok.sort(); print(" ".join(v for _, v in ok[:3]))
PYE
)
say "3 단계 고른 판 (L000·0016·0019 < 10 %, 발산 없음, 평균 작은 순 셋): [$SEL]"
if [ -n "$SEL" ]; then
  L=/dev/shm/dkta/an_lists/d1b.txt; : > $L; DONEI=""
  for v in $SEL; do A=$(grep -m1 "^$v 0016 " /dev/shm/dkta/an_lists/d1a.txt | cut -d' ' -f3-)
    for t in $(seq -f %04g 0 19); do case $t in 0016|0019) continue;; esac; echo "$v $t $A" >> $L; done
    i=${v:0:2}; case " $DONEI " in *" $i "*) continue;; esac; DONEI="$DONEI $i"
    I=$(grep -m1 "^$v 0016 " /dev/shm/dkta/an_lists/d1a.txt | grep -o -- "--integ be\|--ga hht --rho 0\.[0-9]")
    echo "fi$i ficus $B --det $I $OK $T --newton 2 --cg 20" >> $L
  done
  say "3 단계 목록 $(wc -l < $L) 줄"
  AN_CARDS="1 2 3 4 5 6" AN_PAR=6 AN_GAP=5 bash exe/fem_an_pool.sh d1 $L > $M/d1b_pool.log 2>&1
fi
{ echo "# d1 ($(cat .done)) -- 절 AN.21 띠 고착·비덩어리 장벽 헤시안; 3 단계 고른 판 [$SEL]"; echo
  echo "## 판정 (결정적, --mv2 --gq_oct 0.1, εN 1e-2 · εP 1e-1, lego·낙하 1x5, ficus 2x20; 이름 = 적분기 be·h8·h7 + 바닥 s1 (띠 밖 0)·s2 (띠 밖 정답 가중)·s1p (+점별) + 헤시안 l 덩어리·f 비덩어리; no = κ 없음, cv = 8x40 수렴 예산)"
  $PY exe/fem_an_tol_table.py $R
  for d in $R/*_drop1; do v=$(basename $d); echo "- ${v%_drop1} 낙하: $(tail -1 $d/drop_ana.txt 2>/dev/null | cut -c1-200)"; done
  echo; echo "## 단독 시간 (카드 0, 비결정; lw = 최악 허용치 0 10 프레임, lwc = 상한 훑기, lr = 공통 허용치 80 프레임)"; cat $RT/run.out; } > $M/d1_table.md 2>&1
say "[d1 끝] $(date '+%F %T')"
echo "[d1 끝] $(date '+%F %T') 고른 판 [$SEL]" > $M/d1.END
rclone copy $M $MR -q --include "d1*"
