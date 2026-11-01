"""절 AN.err 표: 묶음 디렉토리들 (R2 마운트) 의 <판>_<궤적> 에서 평균 오차·발산·최대 |g|/|g0| (접촉 프레임만)·걸음 0 횟수, --en 이면 충격 창 몸 높이·입자 탄성 비, 낙하 블록은 drop_ana.
사용: python exe/fem_anerr_table.py <묶음 디렉토리> ..."""
import glob
import json
import os
import re
import sys

import numpy as np

rows = {}
for D in sys.argv[1:]:
    for p in sorted(glob.glob(os.path.join(D, "*_*"))):
        if not os.path.isdir(p) or not os.path.exists(os.path.join(p, "fem.jsonl")):
            continue
        nm = os.path.basename(p)
        v, t = nm.rsplit("_", 1)
        R = []
        for l in open(os.path.join(p, "fem.jsonl")):
            try:
                R.append(json.loads(l))
            except json.JSONDecodeError:                                               # 끊긴 줄 (노드 디스크 가득 참 때)
                pass
        if not R:
            continue
        s = json.load(open(os.path.join(p, "summary.json"))) if os.path.exists(os.path.join(p, "summary.json")) else {}
        e = [r["err"] for r in R]
        grc = [r.get("gr", 0) for r in R if (r.get("stuck") or 0) > 0]
        st0 = sum(int(r.get("st0") or 0) for r in R)
        o = dict(err=100 * float(np.mean(e)) if e else float("nan"), n=len(e), div=s.get("diverged"), gr=max(grc) if grc else None, st0=st0)
        en = [r.get("en") for r in R if r.get("en")]
        if en and "psi_gt" in en[0]:
            hc = np.array([q["hc"] for q in en]); hg = np.array([q["hc_gt"] for q in en])
            ps = np.array([q["psi"] for q in en]); pg = np.array([q["psi_gt"] for q in en])
            hr = np.array([q["hr"] for q in en]); hrg = np.array([q["hr_gt"] for q in en])
            ke = np.array([q["ke"] for q in en]); kg = np.array([q.get("ke_gt", np.nan) for q in en])
            i0 = int(np.argmax(pg))                                                    # 정답 입자 탄성 최대 프레임 (첫 충격 최대 압축)
            w = slice(i0, min(i0 + 15, len(en)))
            o.update(i0=i0, dhr=100 * float(np.mean(hr[w] - hrg[w])), dhc=100 * float(np.mean(hc[w] - hg[w])), psr=float(np.mean(ps[w]) / max(np.mean(pg[w]), 1e-9)), ker=float(np.mean(ke[w]) / max(np.nanmean(kg[w]), 1e-9)),
                     ps_end=float(np.mean(ps[-10:])), pg_end=float(np.mean(pg[-10:])), ke_end=float(np.mean(ke[-10:])), kg_end=float(np.nanmean(kg[-10:])),
                     dhc_end=100 * float(hc[-1] - hg[-1]))
        da = os.path.join(p, "drop_ana.txt")
        if os.path.exists(da):
            m = re.search(r"반발 계수 ([\d.]+), 튀어 오름 ([-\d.]+) mm, 최대 압축 ε_zz ([-\d.]+).*복원 시간 ([\d.na]+) ms", open(da).read().split("\n")[-2] if open(da).read().strip() else "")
            for l in open(da):
                if l.startswith(v + ":"):
                    m = re.search(r"반발 계수 ([\d.]+), 튀어 오름 ([-\d.]+) mm, 최대 압축 ε_zz ([-\d.]+).*복원 시간 ([\d.na]+) ms", l)
            if m:
                o.update(e_rest=float(m.group(1)), bounce=float(m.group(2)), ezz=float(m.group(3)), trec=m.group(4))
        rows[(v, t)] = o
vs = sorted({k[0] for k in rows})
ts = sorted({k[1] for k in rows})
print("| 판 | " + " | ".join(ts) + " | lego 평균 | <10% |")
print("|---|" + "---|" * (len(ts) + 2))
for v in vs:
    c, le = [], []
    for t in ts:
        o = rows.get((v, t))
        if o is None:
            c.append("")
            continue
        if t == "drop1" and "e_rest" in o:
            c.append(f"e {o['e_rest']:.3f} / {o['bounce']:.1f} mm" + (" 발산 %s" % o["div"] if o["div"] else ""))
            continue
        c.append(f"{o['err']:.2f}" + (f" (x{o['div']})" if o["div"] else ""))
        if t != "drop1" and not o["div"]:
            le.append(o["err"])
    print(f"| {v} | " + " | ".join(c) + f" | {np.mean(le) if le else float('nan'):.2f} | {sum(x < 10 for x in le)}/{len(le)} |")
print()
print("충격 창 (정답 입자 탄성 최대 프레임부터 15 프레임): 몸 법선 높이 범위 차 (cm, 우리 − 정답) / 중심 높이 차 (cm) / 입자 탄성 비 / 운동 에너지 비 (우리 ÷ 정답); 끝 10 프레임 탄성·운동 J 우리 / 정답; 끝 중심 높이 차 (cm); 접촉 프레임 최대 |g|/|g0|, 걸음 0 횟수")
for (v, t), o in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    if "dhr" in o:
        print(f"{t} {v} (창 f{o['i0']}): {o['dhr']:+.1f} / {o['dhc']:+.1f} / {o['psr']:.2f} / {o['ker']:.2f}; 끝 {o['ps_end']:.0f}/{o['pg_end']:.0f} J, {o['ke_end']:.1f}/{o['kg_end']:.1f} J; {o['dhc_end']:+.1f} cm; 오차 {o['err']:.2f}; gr {o['gr'] if o['gr'] is None else '%.1e' % o['gr']}, st0 {o['st0']}")
    elif t == "drop1":
        print(f"{t} {v}: 오차 {o['err']:.2f} 프레임 {o['n']} 발산 {o['div']} gr {o['gr'] if o['gr'] is None else '%.1e' % o['gr']} st0 {o['st0']} " + (f"e {o.get('e_rest')} 튀어오름 {o.get('bounce')} ε {o.get('ezz')} 복원 {o.get('trec')}" if "e_rest" in o else ""))
