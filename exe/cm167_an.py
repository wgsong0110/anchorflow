"""절 CM167 분석 (CPU, numpy): 판 디렉토리 <run>/cm167.jsonl (fem_cm167: 윗덩어리 뉴턴 반복별 기록) + nl.jsonl (fem_nlwhy: 전체 |g|·PCG·보폭) 에서
프레임마다 한 줄: 덩어리 질량 · 순 잔차 Σg_z/(M g) (반복 0·5·10·20·40·마지막) · 마지막 10 반복 감소비 · 부호 바뀜 · 꼬리<->꼭짓점 바뀜 점 (반복 20·40·마지막) · X1 = Σm(y−Ũ)_z/(Mτ²) (반복 0·10·마지막) · 전체 |g|² 감소비 · 선탐색 보폭 · PCG.
사용: python3 exe/cm167_an.py <run> [--tau2 4.39e-4]"""
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--tau2", type=float, default=4.39e-4); a = ap.parse_args()
G = 9.8
nl = {}
p = os.path.join(a.run, "nl.jsonl")
if os.path.exists(p):
    for l in open(p):
        d = json.loads(l); nl[d["f"]] = d
print("f | M kg | Σg_z/(Mg) it 0·5·10·20·40·끝 | 끝/it49 | 부호 바뀜 (it>=20) | 꼬리↔꼭짓점 바뀜 it 20·40·끝 | 꼬리 점 끝 | X1 it 0·10·끝 | |g|² 끝/it49 | 선탐색 보폭 최소 | PCG nP·도달·Prel (it 50~59)")
for l in open(os.path.join(a.run, "cm167.jsonl")):
    r = json.loads(l); its = r["its"]; n = len(its)
    if n < 22: continue
    M = its[0]["M"]; gz = np.array([o["gz"] for o in its]) / (M * G); e = n - 1
    d = np.diff(gz[20:]); sc = int((np.sign(d[1:]) != np.sign(d[:-1])).sum())
    X1 = [its[i]["ymu"] / M / a.tau2 for i in (0, 10, e)]
    g2 = [o["g2"] for o in its]
    s = nl.get(r["f"], {}).get("steps", [{}])[0].get("its", [])
    ls = min([o.get("ls", 1) for o in s if o.get("ls") is not None] or [0]); nP = np.mean([o.get("nP", 0) for o in s[50:60]] or [0]); pr = np.mean([o.get("Preach", 0) for o in s[50:60]] or [0]); prel = np.mean([(o.get("Prel") or 0) for o in s[50:60]] or [0])
    print(f"f{r['f']} | {M:.1f} | {[round(float(gz[i]), 2) for i in (0, 5, 10, 20, 40, e)]} | {gz[e]/gz[e-10]:.3f} | {sc} | {[its[i]['pl'][5] for i in (20, 40, e)]} | {its[e]['pl'][1]} | {[round(x, 1) for x in X1]} | {g2[e]/g2[e-10]:.3f} | {ls:.2f} | {nP:.0f} {pr:.2f} {prel:.4f}")
