#!/bin/bash
# 절 AB 1 차: 연기 시험 (카드 1, lego N192 5 프레임 클램프 5: 기존 / 각 방식 β 0 (기존과 같아야) / β 1 (도는지)) -> 통과하면 1 차 스윕 (fem_ab_jobs.py, 카드 1·2 x 일꾼 3, 비측정) -> 표
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
OL=$W/lbours/fem_ab_lego; OF=$W/lbours/fem_ab_ficus; mkdir -p $OL $OF
LB="--mode lag --opt --mass tet --no_dump --floor vel --rstat --frames 5 --lim_sr 5 --N 192"
EX_ROOT=fem_ab_lego bash exe/fem_c_run.sh s_T 1 $LB < /dev/null
for f in bar nbar navg polar; do for b in 0 1; do EX_ROOT=fem_ab_lego bash exe/fem_c_run.sh s_${f}_b$b 1 $LB --fupd $f --fupd_b $b < /dev/null; done; done
python - > $OL/smoke.txt 2>&1 <<'PY'
import json
L = "/home/dkta/work/lbours/fem_ab_lego"
def e(t):
    try:
        return [json.loads(l)["err"] for l in open(f"{L}/{t}/fem.jsonl")]
    except Exception as ex:
        return str(ex)
b = e("s_T")
ok = isinstance(b, list) and len(b) == 5
print("기존", b)
for f in ("bar", "nbar", "navg", "polar"):
    z, o = e(f"s_{f}_b0"), e(f"s_{f}_b1")
    rel = max(abs(p - q) / max(abs(q), 1e-12) for p, q in zip(z, b)) if isinstance(z, list) and len(z) == 5 and ok else float("inf")
    print(f, "β0 상대 차 최대", rel, "| β1", o)
    ok = ok and rel < 1e-3 and isinstance(o, list) and len(o) == 5
print("SMOKE_PASS" if ok else "SMOKE_FAIL")
PY
cat $OL/smoke.txt
rclone copy $OL r2:storage/result/anchorflow/lbours/fem_ab_lego -q
grep -q SMOKE_PASS $OL/smoke.txt || { echo "[중단] 연기 시험 실패"; exit 1; }
for k in 0 1 2 3 4 5; do bash exe/fem_ab_gpu.sh $((1 + k % 2)) $k 6 exe/fem_ab_jobs.py > $OL/w1_$k.log 2>&1 < /dev/null & done
wait
python exe/fem_ab_table.py $OL/ab_l_* > $OL/table1.md
python exe/fem_ab_table.py $OF/ab_f_* > $OF/table1.md
cat $OL/table1.md $OF/table1.md
rclone copy $OL r2:storage/result/anchorflow/lbours/fem_ab_lego -q
rclone copy $OF r2:storage/result/anchorflow/lbours/fem_ab_ficus -q
echo AB1_DONE $(date +%T)
