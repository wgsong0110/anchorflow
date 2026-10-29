#!/bin/bash
# 절 U 1 차: (같은 잡의 발산 노드 진단 fem_dv.sh 가 끝나고 7 장이 빌 때까지 대기) -> 연기 시험 (3 프레임 + 노드 에너지 미분 검산) -> 통과하면
#   GPU 0~6 에 나눠 동시에 스윕 (exe/fem_u_jobs.py, 비측정) -> 표·ficus 주기.  속도는 2 차 (단독) 값을 쓴다.
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work
export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
BT=${U_BETA:-0}
echo "[커밋] $(git log -1 --oneline) $(date +%T) β $BT"
mkdir -p $W/lbours/fem_u_lego $W/lbours/fem_u_ficus
until [ -f $W/lbours/fem_dv_lego/DV_DONE ] || [ -f $W/lbours/fem_dv_lego/DV_SMOKE_FAIL ]; do sleep 60; done
echo "[다른 작업 (fem_dv) 끝 확인] $(date +%T)"
while [ -n "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 500')" ]; do echo "[대기] GPU 사용 중 $(date +%T)"; sleep 60; done
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
LB="--mode lag --opt --mass tet --no_dump --floor vel --rstat --frames 3 --N 192"
FB="--mode lag --opt --mass tet --no_dump --dump_pred --data $FD --tag F_pg_v0 --frames 3 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --rstat --N 640"
S="bash exe/fem_ex_run.sh"
export EX_NOWAIT=1
SB=$( [ "$BT" = 0 ] && echo "" || echo "_b$BT" )
( EX_ROOT=fem_u_lego $S u_smoke${SB}_l 0 $LB --nint --nint_beta $BT --nint_check 3 ) > /dev/null 2>&1 &
( EX_ROOT=fem_u_lego $S u_smoke${SB}_lL 1 $LB --nint --nint_beta $BT --nint_check 3 --lim 0.3 --lim_mode box ) > /dev/null 2>&1 &
( EX_ROOT=fem_u_lego $S u_smoke${SB}_l_b20 2 $LB --nint --nint_beta 0.2 --nint_check 3 ) > /dev/null 2>&1 &
( EX_ROOT=fem_u_lego $S u_smoke${SB}_l_ng 3 $LB --nint --nint_beta $BT --no_graph ) > /dev/null 2>&1 &
( EX_ROOT=fem_u_ficus $S u_smoke${SB}_f 4 $FB --nint --nint_beta $BT --nint_check 3 ) > /dev/null 2>&1 &
( EX_ROOT=fem_u_lego $S u_smoke${SB}_old 5 $LB ) > /dev/null 2>&1 &
wait
unset EX_NOWAIT
bad=0
for d in $W/lbours/fem_u_lego/u_smoke${SB}_* $W/lbours/fem_u_ficus/u_smoke${SB}_*; do
  if [ -f $d/summary.json ]; then echo "[연기 OK] $(basename $d) $(grep -a '롤아웃\]' $d/v_fem.log | cut -c1-150)"; grep -a "명시적 FEM\]\|프레임 [123]/\|U 검산\|경고" $d/v_fem.log | cut -c1-600; else echo "[연기 실패] $(basename $d)"; tail -25 $d/v_fem.log; bad=1; fi
done
# 검산 통과 기준: 방향 미분 (eps 1e-6) 과 노드 성분 상대차 < 1e-2 (커널 float32 vs float64 에너지 중앙 차분)
python - $W/lbours/fem_u_lego/u_smoke${SB}_l $W/lbours/fem_u_lego/u_smoke${SB}_l_b20 $W/lbours/fem_u_ficus/u_smoke${SB}_f <<'PY' || bad=1
import json, sys
ok = True
for d in sys.argv[1:]:
    try:
        R = [json.loads(l) for l in open(d + "/nint_check.jsonl")]
    except Exception as e:
        print("[검산 없음]", d, e); ok = False; continue
    for r in R:
        rd = max(q["rel"] for q in r["dirs"] if q["eps"] == 1e-6); rc = max(q["rel"] for q in r["comp"])
        print(f"[검산] {d.split('/')[-1]} t {r['t'] + 1} β {r['beta']}: 방향 {rd:.2e} 성분 {rc:.2e} Σf {r['f_sum']} Σ|f| {r['f_abs']:.4g}")
        ok = ok and rd < 1e-2 and rc < 1e-2
sys.exit(0 if ok else 1)
PY
rclone copy $W/lbours/fem_u_lego r2:storage/result/anchorflow/lbours/fem_u_lego -q
rclone copy $W/lbours/fem_u_ficus r2:storage/result/anchorflow/lbours/fem_u_ficus -q
[ $bad -eq 0 ] || { echo U1_SMOKE_FAIL; exit 1; }
echo U1_SMOKE_OK $(date +%T)
for G in 0 1 2 3 4 5 6; do bash exe/fem_u_gpu.sh $G exe/fem_u_jobs.py $BT < /dev/null > $W/fem_u1${SB}_g$G.out 2>&1 & done
wait
SF=$( [ "$BT" = 0 ] && echo "" || echo "_b$(python -c "print('%02d' % round($BT*100))")" )
python exe/fem_p_period.py --data $FD --pred $W/lbours/fem_u_ficus/u_f${SF}_*/pred.pt > $W/lbours/fem_u_ficus/period1${SB}.txt 2>&1
python exe/fem_r_table.py $W/lbours/fem_u_lego/u_l${SF}_[oL]_* > $W/lbours/fem_u_lego/table1${SB}.md
python exe/fem_r_table.py $W/lbours/fem_u_ficus/u_f${SF}_[oL]_* > $W/lbours/fem_u_ficus/table1${SB}.md
rclone copy $W/lbours/fem_u_lego r2:storage/result/anchorflow/lbours/fem_u_lego -q
rclone copy $W/lbours/fem_u_ficus r2:storage/result/anchorflow/lbours/fem_u_ficus -q
echo U1_DONE $(date +%T)
