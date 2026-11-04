#!/bin/bash
# 절 AN.20 묶음 (카드 0 단독, gpu0.lock, 판마다 카드 0 의 다른 계산 프로세스 0 확인; 비결정 = 시간, 결정적 = 판정):
#  A. 결과 확인 (결정적 ficus 10 프레임 해시) + 커널별 그래프 시간 (kbench)
#  B. 최악 (허용치 0 = 상한까지) 시간: 원래 판 / --mv2 / --mg_occ -- ficus 20 프레임 2x10, lego 10 프레임 1x5 (2 회)
#  C. ficus 상한 훑기 (최악 20 프레임, 판 MGV + SWX (기본 --gq_oct 0.1)): 최대 시간 16.7 ms 안 가장 큰 뉴턴 x PCG (같으면 뉴턴 큰 것) 둘
#  D. 고른 상한 + 공통 허용치 (εN 1e-2, εP 1e-1) ficus 125 프레임: 비결정 시간 + 결정적 bej·bejo (κ 0.1) 주기; lego L000 80 프레임 1x5 (시간 + 결정적 오차)
cd "$(dirname "$0")/.." || exit 1
PY=/home/dkta/.conda/envs/af/bin/python
export PYTHONUTF8=1 OMP_NUM_THREADS=8 PATH=$HOME/bin:/home/dkta/.conda/envs/af/bin:$PATH
export CUPY_CACHE_DIR=/dev/shm/dkta/cache/cupy_mg XDG_CACHE_HOME=/dev/shm/dkta/cache
NM=${1:-mg1}; R=/dev/shm/dkta/lbours/fem_an/$NM; M=/dev/shm/dkta/lbours/fem_an/marks; mkdir -p $R $M $CUPY_CACHE_DIR
RR=r2:storage/result/anchorflow/lbours/fem_an/$NM
SD=/dev/shm/dkta/anerr_data; FD=$SD/F_pg_v0.pt
U0=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
others0() { nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$U0"; }
exec 9>/home/dkta/work/gpu0.lock; flock 9
echo "잠금 $(date +%T) $(cat .done)" | tee -a $R/run.out
B="--cell 1 --mg_levels 4 --mg_nc 4 --ls_n 10 --descent --K 2 --mg_wreuse --mg_lreuse --integ be"
MGV=${MGV:---mv2}
SWX=${SWX---gq_oct 0.1}                                                   # 상한 훑기·시간 판 추가 인자 (기본 = 판정 판 bejo 의 점유 가중 κ 0.1)
T="--tolN 1e-2 --tolP 1e-1"; T0="--tolN 0 --tolP 0"
LG="--data $SD/L000.pt --floor vel --tag L000"
FI="--data $FD --floor none --cfg_bc --region_density --pg_vdamp --tag ficus"
run() { local v=$1; shift; local O=$R/$v; mkdir -p $O
  while [ "$(others0)" -gt 0 ]; do sleep 5; done
  echo "시작 때 카드 0 의 다른 프로세스 $(others0) 개 $(date +%T)" > $O/card.txt
  CUDA_VISIBLE_DEVICES=0 timeout ${MG_TMO:-1800} $PY -u exe/fem_an_rt_sim.py --out $O $B "$@" 9>&- < /dev/null > $O/log.txt 2>&1
  echo "끝 때 카드 0 의 다른 프로세스 $(others0) 개 $(date +%T)" >> $O/card.txt
  [ -f $O/pred.pt ] && grep -q ficus <<< "$*" && $PY exe/fem_p_period.py --data $FD --pred $O/pred.pt > $O/period.txt 2>&1
  echo "$v $(date +%T) $(grep -a '롤아웃\|rror\|발산' $O/log.txt | tail -2 | cut -c1-200 | tr '\n' ' ') $($PY -c "import json;d=json.load(open('$O/summary.json'));print('sha',d['x_sha1'],'mean',round(d.get('ms_mean') or -1,2),'max',round(d.get('ms_max') or -1,2),'p90',round(d.get('ms_p90') or -1,2),'nP',(d.get('it_nP') or {}).get('mean'),'nPmax',(d.get('it_nP') or {}).get('max'),'nN',(d.get('it_nN') or {}).get('mean'))" 2>/dev/null) $(grep -a '대비' $O/period.txt 2>/dev/null | cut -c1-160)" | tee -a $R/run.out
  grep -q Error $O/log.txt && tail -20 $O/log.txt
  rclone copy $O $RR/$v -q; }
if [ -z "${MG_SKIP_AB:-}" ]; then
run fd_mv2 $FI --frames 10 $T0 --newton 2 --cg 10 --det --mv2
run fd_occ $FI --frames 10 $T0 --newton 2 --cg 10 --det --mg_occ
run kb_fi $FI --frames 6 --kbench 6 $T0 --newton 2 --cg 10 --mg_occ --mg_fuse 3
run kb_lg $LG --frames 6 --kbench 6 $T0 --newton 1 --cg 5 --mg_occ --mg_fuse 3
for rep in 1 2; do
  run fw_base_r$rep $FI --frames 20 $T0 --newton 2 --cg 10
  run fw_mv2_r$rep $FI --frames 20 $T0 --newton 2 --cg 10 --mv2
  run fw_occ_r$rep $FI --frames 20 $T0 --newton 2 --cg 10 --mg_occ
  run lw_base_r$rep $LG --frames 10 $T0 --newton 1 --cg 5
  run lw_mv2_r$rep $LG --frames 10 $T0 --newton 1 --cg 5 --mv2
  run lw_occ_r$rep $LG --frames 10 $T0 --newton 1 --cg 5 --mg_occ
done
fi
CAPS=${MG_CAPS:-2x10 2x14 2x16 2x18 2x20 2x24 3x10 3x12 3x14 3x16 4x10 4x12 1x30 1x40}
for cap in $CAPS; do n=${cap%x*}; c=${cap#*x}
  run fwc_$cap $FI --frames 20 $T0 --newton $n --cg $c $MGV $SWX
done
SEL=$($PY - $R $CAPS <<'PYE'
import json, sys
R = sys.argv[1]; ok = []
for cap in sys.argv[2:]:
    try: d = json.load(open(f"{R}/fwc_{cap}/summary.json"))
    except Exception: continue
    n, c = map(int, cap.split("x"))
    if d.get("ms_max") and d["ms_max"] <= 16.7: ok.append((n * c, n, cap))
ok.sort(reverse=True)
print(" ".join(o[2] for o in ok[:2]) if ok else "2x10")
PYE
)
echo "고른 ficus 상한 (최악 최대 16.7 ms 안, 뉴턴 x PCG 큰 순 둘) = $SEL" | tee -a $R/run.out
for cap in $SEL; do n=${cap%x*}; c=${cap#*x}
  run fic_$cap $FI --frames 125 $T --newton $n --cg $c $MGV --dump
  run fico_$cap $FI --frames 125 $T --newton $n --cg $c $MGV --gq_oct 0.1 --dump
  run fico2_$cap $FI --frames 125 $T --newton $n --cg $c $MGV --gq_oct 0.1
  run ficd_bej_$cap $FI --frames 125 $T --newton $n --cg $c $MGV --det --dump
  run ficd_bejo_$cap $FI --frames 125 $T --newton $n --cg $c $MGV --det --gq_oct 0.1 --dump
done
run lg_base $LG --frames 80 $T --newton 1 --cg 5
run lg_mgv $LG --frames 80 $T --newton 1 --cg 5 $MGV
run lgd_mgv $LG --frames 80 $T --newton 1 --cg 5 $MGV --det
flock -u 9; exec 9>&-
{ echo "# $NM ($(cat .done)), 판 $MGV, 고른 ficus 상한 $SEL"; echo; cat $R/run.out 2>/dev/null; } > $M/$NM.md
echo "[끝] $(date '+%F %T') 고른 상한 $SEL" > $M/$NM.END
rclone copy $M r2:storage/result/anchorflow/lbours/fem_an/marks -q --include "$NM.*"; rclone copy $R/run.out $RR -q
