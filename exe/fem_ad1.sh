#!/bin/bash
# 절 AD 1 차: 연기 시험 (카드 3, lego N192 5 프레임 클램프 5: 기준 / merge k0 (희소 없음 -> 묶음 커널 경로가 기준과 같아야) / 각 변형) -> 통과하면 스윕 (fem_ad_jobs.py, 카드 3·4 x 일꾼 3, 비측정) -> 주기·표·R2
cd "$(dirname "$0")/.." || exit 1
W=/home/dkta/work; export PYTHONUTF8=1
set +u; source /tools/anaconda3/etc/profile.d/conda.sh; conda activate af; export PATH=$HOME/bin:$PATH
echo "[커밋] $(git log -1 --oneline) $(date +%T)"
OL=$W/lbours/fem_ad_lego; OF=$W/lbours/fem_ad_ficus; mkdir -p $OL $OF
LB="--mode lag --opt --mass tet --no_dump --floor vel --rstat --frames 5 --lim_sr 5 --N 192"
R="bash exe/fem_c_run.sh"
EX_ROOT=fem_ad_lego $R s_ref 3 $LB < /dev/null
EX_ROOT=fem_ad_lego $R s_mg0 3 $LB --sp merge --sp_k 0 --sp_chk 1,3,5 < /dev/null
EX_ROOT=fem_ad_lego $R s_cb0 3 $LB --sp cube --sp_k 0 --sp_chk 1,3,5 < /dev/null
for v in "mg2 --sp merge --sp_k 2" "cb2 --sp cube --sp_k 2" "bl2 --sp blend --sp_k 2" "st2 --sp stiff --sp_k 2" "ml01 --sp mlow --sp_mlow 0.1"; do
  set -- $v; t=$1; shift
  EX_ROOT=fem_ad_lego $R s_$t 3 $LB "$@" --sp_chk 1,3,5 < /dev/null
done
python - > $OL/smoke.txt 2>&1 <<'PY'
import json
L = "/home/dkta/work/lbours/fem_ad_lego"
def e(t):
    try:
        return [json.loads(l)["err"] for l in open(f"{L}/{t}/fem.jsonl")]
    except Exception as ex:
        return str(ex)
b = e("s_ref")
ok = isinstance(b, list) and len(b) == 5
print("기준", b)
for t in ("s_mg0", "s_cb0"):
    z = e(t)
    rel = max(abs(p - q) / max(abs(q), 1e-12) for p, q in zip(z, b)) if isinstance(z, list) and len(z) == 5 and ok else float("inf")
    print(t, "기준 대비 상대 차 최대", rel)
    ok = ok and rel < 1e-3
for t in ("s_mg2", "s_cb2", "s_bl2", "s_st2", "s_ml01"):
    z = e(t)
    print(t, z)
    ok = ok and isinstance(z, list) and len(z) == 5
print("SMOKE_PASS" if ok else "SMOKE_FAIL")
PY
cat $OL/smoke.txt
grep -a "절 AD\|롤아웃\]\|Error\|error" $OL/s_*/v_fem.log | cut -c1-400
for t in s_ref s_mg2; do echo "$t 시작 $(stat -c %y $OL/$t/cmd.txt | cut -c12-19) 끝 $(stat -c %y $OL/$t/summary.json 2>/dev/null | cut -c12-19)"; done
rclone copy $OL r2:storage/result/anchorflow/lbours/fem_ad_lego -q
grep -q SMOKE_PASS $OL/smoke.txt || { echo "[중단] 연기 시험 실패"; exit 1; }
echo "[스윕 시작] $(date +%T)"
for k in 0 1 2 3 4 5; do bash exe/fem_ad_gpu.sh $((3 + k % 2)) $k 6 exe/fem_ad_jobs.py > $OL/w1_$k.log 2>&1 < /dev/null & done
wait
FD=$W/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt
P=$(ls $OF/ad_f_*/pred.pt 2>/dev/null); [ -n "$P" ] && python exe/fem_p_period.py --data $FD --pred $P > $OF/period1.txt 2>&1
python exe/fem_ad_table.py $OL/ad_l_* > $OL/table1.md
python exe/fem_ad_table.py $OF/ad_f_* > $OF/table1.md
cat $OL/table1.md $OF/table1.md
rclone copy $OL r2:storage/result/anchorflow/lbours/fem_ad_lego -q
rclone copy $OF r2:storage/result/anchorflow/lbours/fem_ad_ficus -q
echo AD1_DONE $(date +%T)
