"""wolffix 집계 (11-15, 읽기만): 한 wolf 판 디렉토리 -> 오차·시간·높이·튐·뒤집힘·덜 풀린 걸음 (wfx_an.json + 화면).
사용: python exe/wfx_an.py <판 디렉토리> <정답 궤적 디렉토리 gt_wolf>
 - 튐: 정답과 같은 인덱스 입자 거리 > 0.1 m 수 (far), 정답 입자 x 최소 (0.5 백분위) 보다 5 cm 넘게 −x 로 나간 입자 수 (left), det F ≤ 0 입자 수 (inv).
 - 높이: fem.jsonl hp[0]/hg[0] − 0.48 (mm, 절 CM117 정의). 프레임 끝 움직임: 프레임 사이 평균 변위 (mm, 우리 대 정답) -- 마지막 무너짐 지표.
 - 덜 풀린 걸음: nl.jsonl 걸음 끝 g2end (‖g‖²/F²) > 0.1, 장벽 보폭 < 1."""
import json, os, sys
import h5py
import numpy as np
import torch

D, GT = sys.argv[1], sys.argv[2]
o = dict(run=os.path.basename(D.rstrip("/")))
try:
    sm = json.load(open(os.path.join(D, "summary.json")))
    o.update(err=sm.get("err_mean", sm.get("err")), ms=sm.get("ms_mean", sm.get("ms")), frames=sm.get("frames_done", sm.get("n_frames")))
except Exception as e:
    o["summary_err"] = repr(e)
lg = os.path.join(D, "log.txt")
if os.path.exists(lg):
    for q in open(lg, errors="ignore"):
        if "[롤아웃]" in q:
            o["rollout"] = q.strip()[:300]
F = [json.loads(q) for q in open(os.path.join(D, "fem.jsonl")) if q.strip()] if os.path.exists(os.path.join(D, "fem.jsonl")) else []
hp = {int(r["t"]) + 1: (1000 * (r["hp"][0] - 0.48), 1000 * (r["hg"][0] - 0.48)) for r in F if "hp" in r and "hg" in r}
o["h"] = {f: [round(hp[f][0]), round(hp[f][1])] for f in (20, 30, 40, 45, 50) if f in hp}
o["ms_f"] = [round(r.get("ms", 0), 2) for r in F]
o["nn_f"] = [r.get("nn") for r in F]
NL = [json.loads(q) for q in open(os.path.join(D, "nl.jsonl")) if q.strip()] if os.path.exists(os.path.join(D, "nl.jsonl")) else []
bad = []
for r in NL:
    for s_ in r["steps"]:
        ls = [i.get("ls", 1.0) for i in s_["its"] if i.get("r") is not None]
        if s_.get("g2end", 0) > 0.1 or (ls and min(ls) < 1.0 and min(ls) > 0):
            bad.append([r["f"], s_["k"], round(s_.get("F2", 0), -6), round(s_.get("g2end", 0), 3), min(ls) if ls else None])
o["bad_steps"] = bad
pp = os.path.join(D, "pred.pt")
if os.path.exists(pp):
    d = torch.load(pp, map_location="cpu", weights_only=False)
    X = d["pred"].double(); FF = d.get("F")
    nf = X.shape[0]
    rows = []
    for t in range(nf):
        fn = os.path.join(GT, "sim_%010d.h5" % (t + 1))
        if not os.path.exists(fn):
            break
        with h5py.File(fn) as h:
            g = torch.from_numpy(np.array(h["x"])).double()
        n = min(g.shape[0], X.shape[1]); x = X[t, :n]; g = g[:n]
        dist = (x - g).norm(dim=1)
        gx = torch.quantile(g[:, 0].float(), 0.005).item()
        inv = int((torch.linalg.det(FF[t, :n].double()) <= 0).sum()) if FF is not None else -1
        mv = float((X[t, :n] - X[t - 1, :n]).norm(dim=1).mean() * 1000) if t > 0 else 0.0
        rows.append(dict(f=t + 1, far=int((dist > 0.1).sum()), left=int((x[:, 0] < gx - 0.05).sum()), inv=inv, xmin=round(torch.quantile(x[:, 0].float(), 0.005).item(), 3), gxmin=round(gx, 3), mv=round(mv, 2),
                         err=round(float(dist.mean() * 100), 3)))
    G0 = []
    for t in range(1, len(rows)):
        with h5py.File(os.path.join(GT, "sim_%010d.h5" % (t + 1))) as h, h5py.File(os.path.join(GT, "sim_%010d.h5" % t)) as h0:
            G0.append(float(np.linalg.norm(np.array(h["x"]) - np.array(h0["x"]), axis=1).mean() * 1000))
    for t in range(1, len(rows)):
        rows[t]["gmv"] = round(G0[t - 1], 2)
    o["rows"] = rows
    o["inv_frames"] = [r["f"] for r in rows if r["inv"] > 0]
    o["far50"] = rows[-1]["far"] if rows else None; o["left50"] = rows[-1]["left"] if rows else None
    o["far_max"] = max(r["far"] for r in rows) if rows else None
    o["mv4050"] = round(float(np.mean([r["mv"] for r in rows if 40 <= r["f"] <= 50])), 2) if rows else None
    o["gmv4050"] = round(float(np.mean([r.get("gmv", 0) for r in rows if 40 <= r["f"] <= 50])), 2) if rows else None
json.dump(o, open(os.path.join(D, "wfx_an.json"), "w"))
print(f"[{o['run']}] err {o.get('err')} ms {o.get('ms')} | h {o['h']} | far50 {o.get('far50')} left50 {o.get('left50')} far_max {o.get('far_max')} | inv 프레임 {o.get('inv_frames')} | "
      f"끝 움직임 f40~50 {o.get('mv4050')} (정답 {o.get('gmv4050')}) mm/프레임 | 덜 풀린 걸음 {len(bad)}: {bad[:12]}")
for r in o.get("rows", []):
    if r["f"] >= 30 or r["inv"] > 0:
        print("  ", r)
