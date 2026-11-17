#!/bin/bash
# [stabfinal] ship i-PG 수정판 대표 N 을 64 로 (사용자 지시: N1 은 바닥 관통 30.7 %L·부피 폭주로 제외): N64 덤프·렌더·시각 -> i-PG 자리로 an·extra (ipg 항목만 바꿈). stabfinal_boot 의 단계 ipgpost.
set -u
cd $(dirname $0)/..
S=${1:-ship_plastic_lego}; T_=${S1115_TAG:-}; N=${SF_IPGN:-${T_#bcp}}; N=${N%%[!0-9]*}; N=${N:-64}; ST=/home/dkta/work/stab; D=$ST/$S
EV="--ipgr_env AF_FD_EPS=1e-5,AF_LS_REJECT=1,AF_IPG_GPU=1,AF_IPG_FAST=1${SF_IPGENV_ADD:-,AF_IPG_MATRESTORE=1} --onerep 1 --timeout 43200"
RXR=r2:storage/result/anchorflow/lbours/stabfinal/res/$S
rm -f $D/res/ipg_N* $D/res/ipgr_N*            # [bca0] 이전 판 (바닥 노드 가속도 고치기 전) 결과는 이 인스턴스 분석에서 뺌
python3 - <<PY
import json
p=json.load(open("$D/plan_ipgr.json")); p["N_min_rule"]=p.get("N_min"); p["N_min"]=$N; p["note"]="i-PG 바닥 노드 가속도 수정판 (AF_IPG_BCA0) N$N"
json.dump(p,open("$D/plan_ipgr.json","w"))
q=dict(p); q["method"]="ipg"; json.dump(q,open("$D/plan_ipg.json","w"))
PY
cp $D/vis/vis.json $D/vis/vis_before_N64.json
python -u exe/stab1115_drv.py --scene $S --par 4 --stages vis --methods ipgr $EV > /root/sf/post64_vis.log 2>&1
echo "[vis] $(tail -2 /root/sf/post64_vis.log | tr '\n' ' ' | cut -c1-300)"
ln -sfn $D/vis/png_ipgr_N$N $D/vis/png_ipg_N$N; cp $D/vis/ipgr_N$N.npz $D/vis/ipg_N$N.npz; cp $D/vis/ipgr_N$N.mp4 $D/vis/ipg_N$N.mp4
python3 - <<PY
import json
v=json.load(open("$D/vis/vis_before_N64.json")); n=json.load(open("$D/vis/vis.json"))
[v.pop(k) for k in [k for k in v if k.startswith("ipg_N")]]; v["ipg_N$N"]=n["ipgr_N$N"]; json.dump(v,open("$D/vis/vis.json","w"),indent=1); print("[vis ipg]", v["ipg_N$N"])
PY
for w in $(seq 1 240); do rclone lsf $RXR/res/ipgr_N$N.fin >/dev/null 2>&1 && [ -n "$(rclone lsf $RXR/res/ipgr_N$N.fin)" ] && break; sleep 30; done   # main 판 (다른 인스턴스) 기다림
rclone copy $RXR/res $D/res -q --include "ipgr_N$N.*"
for x in npz log fin; do cp $D/res/ipgr_N$N.$x $D/res/ipg_N$N.$x; done
cp $D/an.json $D/an_before_N64.json; cp $D/extra.json $D/extra_before_N64.json
python -u exe/stab1115_an.py --scene $S --out $D/an_new64.json --hextag _f7 > /root/sf/an64.log 2>&1; echo "an rc=$?"
python -c "import pyiqa" 2>/dev/null || pip install -q pyiqa "torch==2.5.1" "torchvision==0.20.1" "numpy<2" 2>&1 | tail -1
python -u exe/stab_extra.py --scene $S --hextag _f7 > /root/sf/extra64.log 2>&1; echo "extra rc=$?"
mv $D/extra.json $D/extra_new64.json; cp $D/extra_before_N64.json $D/extra.json
python3 - <<PY
import json
o=json.load(open("$D/an_before_N64.json")); n=json.load(open("$D/an_new64.json"))
o["methods"]["ipg"]=n["methods"]["ipg"]; json.dump(o,open("$D/an.json","w"),indent=1); print("[an ipg N_min]", n["methods"]["ipg"].get("N_min"))
e=json.load(open("$D/extra.json")); en=json.load(open("$D/extra_new64.json"))
for sec in e:
    if isinstance(e[sec],dict):
        for k in [k for k in e[sec] if k.startswith("ipg_")]: e[sec].pop(k)
for sec in en:
    if isinstance(en.get(sec),dict):
        e.setdefault(sec,{})
        for k,v in en[sec].items():
            if k.startswith("ipg_"): e[sec][k]=v
json.dump(e,open("$D/extra.json","w"),indent=1)
print("[extra ipg]", {sec:{k:v for k,v in e[sec].items() if k.startswith("ipg_")} for sec in e if isinstance(e[sec],dict)})
PY
