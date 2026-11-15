"""절 CM181: 우리 판 (pred.pt·lj.pt) 대 GF 변형 (gf_mpm_cm181 sv/sub_*.npz) -- 같은 입자 (간격 8) 로 손상 분포·질량중심·위치 차.
사용: python exe/wmfrac_cmp.py FEMDIR GFDIR [FEMDIR2 ...]   (GFDIR = .../out/<이름>; torch 필요 -> 인스턴스에서)
"""
import sys, os
import numpy as np, torch

def fem(d, f):
    p = torch.load(f"{d}/pred.pt", map_location="cpu", weights_only=False)["pred"]
    l = torch.load(f"{d}/lj.pt", map_location="cpu", weights_only=False)["logJp"]
    return p, l

FR = (3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100)
gfd = sys.argv[1]
def gfload(f):
    for nm in (f"sub_{f:04d}.npz",):
        fn = f"{gfd}/sv/{nm}"
        if os.path.exists(fn):
            a = np.load(fn); return a["x"], a["lj"], a["al"] == 1
    fn = f"{gfd}/sv/full_{f:04d}.npz"
    if os.path.exists(fn):
        a = np.load(fn); return a["x"][::8], a["lj"][::8], a["al"][::8] == 1
    return None

def stats(x, lj, ok, bins=np.linspace(0, 0.8, 9)):
    ok = ok & np.isfinite(x).all(1)
    h = x[ok, 2]; d = lj[ok] > 0
    hist = []
    for i in range(len(bins) - 1):
        m = (h >= bins[i]) & (h < bins[i + 1])
        hist.append("%.1f" % (100 * d[m].mean()) if m.sum() > 50 else "-")
    return dict(dmg=100 * d.mean(), zc=h.mean(), h995=np.percentile(h, 99.5), cmp=100 * (lj[ok] < -0.2).mean(), strong=100 * (lj[ok] > 0.4).mean(), hist=" ".join(hist))

print("| 판 | 프레임 | 손상 % | 압축 소성 (logJp<-0.2) % | 강한 손상 (>0.4) % | 질량중심 z | 상단 99.5 % z | 높이 0.1 m 구간별 손상 % (0~0.8) | GF 와 평균 위치 차 (m) |")
print("|---|---|---|---|---|---|---|---|---|")
for fd in sys.argv[2:]:
    p, l = fem(fd, 0)
    for f in FR:
        g = gfload(f)
        if g is None:
            continue
        xg, lg, ag = g
        x = p[f - 1][::8].numpy(); lj = l[f - 1][::8].numpy()
        for nm, (xx, ll, oo) in (("GF", (xg, lg, ag)), (os.path.basename(fd), (x, lj, np.ones(len(x), bool)))):
            s = stats(xx, ll, oo)
            dd = "" if nm == "GF" else "%.4f" % np.nanmean(np.linalg.norm(x - xg, axis=1)[ag])
            print(f"| {nm} | f{f} | {s['dmg']:.2f} | {s['cmp']:.1f} | {s['strong']:.2f} | {s['zc']:.3f} | {s['h995']:.2f} | {s['hist']} | {dd} |")
