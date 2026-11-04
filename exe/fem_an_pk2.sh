#!/bin/bash
# 절 AN.15 지속형 판 2 차 (카드 0 단독, gpu0.lock, 판마다 카드 0 의 다른 계산 프로세스 0 확인; 비결정 = 시간, 결정적 = 판정):
#  A. 그래프 판 vs 지속형 판 시간 (2 회): 공통 허용치 εN 1e-2 · εP 1e-1, lego L000 80 프레임 상한 1x5, ficus 125 프레임 상한 2x10; 최악 (허용치 0 = 상한까지) lego 10·ficus 20 프레임
#  B. 지속형 판 ficus 상한 훑기 (허용치 0 = 최악, 20 프레임): 16.7 ms (최대) 안에 드는 가장 큰 상한 고르기
#  C. 고른 상한 + 공통 허용치로 ficus 125 프레임: 결정적 (bej·bejo, 주기) + 비결정 실제 시간
# 출력 /dev/shm/dkta/lbours/fem_an/<묶음> -> R2, 표 marks/<묶음>.md, 끝 표식 marks/<묶음>.END
cd "$(dirname "$0")/.." || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8 PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_pk XDG_CACHE_HOME=/dev/shm/dkta/cache
NM=${1:-pk2}; R=/dev/shm/dkta/lbours/fem_an/$NM; M=/dev/shm/dkta/lbours/fem_an/marks; mkdir -p $R $M $CUPY_CACHE_DIR
RR=r2:storage/result/anchorflow/lbours/fem_an/$NM
SD=/dev/shm/dkta/anerr_data; FD=$SD/F_pg_v0.pt
U0=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
others0() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$U0"; }
exec 9>/home/dkta/work/gpu0.lock; flock 9
echo "잠금 $(date +%T) $(cat .done)"
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mg_lreuse --integ be"
PKX="--pk --pk_loc ${PK_LOC:-99} ${PK_X:-}"
T="--tolN 1e-2 --tolP 1e-1"; T0="--tolN 0 --tolP 0"
LG="--data $SD/L000.pt --floor vel --tag L000"
FI="--data $FD --floor none --cfg_bc --region_density --pg_vdamp --tag ficus"
run() { local v=$1; shift; local O=$R/$v; mkdir -p $O
  while [ "$(others0)" -gt 0 ]; do sleep 5; done
  echo "시작 때 카드 0 의 다른 프로세스 $(others0) 개 $(date +%T)" > $O/card.txt
  CUDA_VISIBLE_DEVICES=0 timeout ${PK_TMO:-1800} $PY -u exe/fem_an_rt_sim.py --out $O $B "$@" 9>&- < /dev/null > $O/log.txt 2>&1
  echo "끝 때 카드 0 의 다른 프로세스 $(others0) 개 $(date +%T)" >> $O/card.txt
  [ -f $O/pred.pt ] && [ "$(grep -c ficus <<< "$*")" -gt 0 ] && $PY exe/fem_p_period.py --data $FD --pred $O/pred.pt > $O/period.txt 2>&1
  echo "$v $(date +%T) $(grep -a '롤아웃\|rror\|발산' $O/log.txt | tail -2 | cut -c1-200 | tr '\n' ' ') $($PY -c "import json;d=json.load(open('$O/summary.json'));print('max',round(d.get('ms_max') or -1,2),'p90',round(d.get('ms_p90') or -1,2),'nP',(d.get('it_nP') or {}).get('mean'),'nN',(d.get('it_nN') or {}).get('mean'))" 2>/dev/null) $(grep -a '대비' $O/period.txt 2>/dev/null | cut -c1-120)" | tee -a $R/run.out
  rclone copy $O $RR/$v -q; }
if [ -z "${PK_SKIP_A:-}" ]; then
for rep in 1 2; do
  run lg_g_r$rep $LG --frames 80 $T --newton 1 --cg 5
  run lg_pk_r$rep $LG --frames 80 $T --newton 1 --cg 5 $PKX
  run fi_g_r$rep $FI --frames 125 $T --newton 2 --cg 10
  run fi_pk_r$rep $FI --frames 125 $T --newton 2 --cg 10 $PKX
  run lgw_g_r$rep $LG --frames 10 $T0 --newton 1 --cg 5
  run lgw_pk_r$rep $LG --frames 10 $T0 --newton 1 --cg 5 $PKX
  run fiw_g_r$rep $FI --frames 20 $T0 --newton 2 --cg 10
  run fiw_pk_r$rep $FI --frames 20 $T0 --newton 2 --cg 10 $PKX
done
fi
for cap in ${PK_CAPS:-1x10 2x5 1x20 2x10 3x10 2x20 3x20 4x20}; do n=${cap%x*}; c=${cap#*x}
  run fiw_pk_$cap $FI --frames 20 $T0 --newton $n --cg $c $PKX
done
SEL=$($PY - $R ${PK_CAPS:-1x10 2x5 1x20 2x10 3x10 2x20 3x20 4x20} <<'PYE'
import json, sys
R = sys.argv[1]; best = None
for cap in sys.argv[2:]:
    try: d = json.load(open(f"{R}/fiw_pk_{cap}/summary.json"))
    except Exception: continue
    n, c = map(int, cap.split("x"))
    if d.get("ms_max") and d["ms_max"] <= 16.7 and (best is None or n * c > best[0]): best = (n * c, cap)
print(best[1] if best else "1x10")
PYE
)
echo "고른 ficus 상한 (최악 최대 16.7 ms 안 가장 큰 뉴턴 x PCG) = $SEL"
n=${SEL%x*}; c=${SEL#*x}
run fic_pk_$SEL $FI --frames 125 $T --newton $n --cg $c $PKX --dump
run ficd_bej_$SEL $FI --frames 125 $T --newton $n --cg $c $PKX --det --dump
run ficd_bejo_$SEL $FI --frames 125 $T --newton $n --cg $c $PKX --det --gq_oct 0.1 --dump
flock -u 9; exec 9>&-
{ echo "# $NM ($(cat .done)), 고른 ficus 상한 $SEL"; echo; cat $R/run.out 2>/dev/null; } > $M/$NM.md
echo "[끝] $(date '+%F %T') 고른 상한 $SEL" > $M/$NM.END
rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q --include "$NM.*"; rclone copy $R/run.out $RR -q
