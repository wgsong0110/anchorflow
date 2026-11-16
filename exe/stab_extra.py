"""[stabfinal] 안정성 추가 지표 (한 장면, 확정 최소 N 의 모든 방법 + 기준 PG N=1024) -> <장면>/extra.json
물리 (정답 없음, 궤적 표본점 X 와 에너지 기록 rows):
  energy_inc = max(0, max_t (E_t - E_0) / |E_0|), E = 운동 + 위치 에너지 (안정 판정과 같은 E; 탄성 에너지는 방법마다 기록 여부가 달라 뺌)
  vol_change = max_t |V_t / V_0 - 1|, V_t = 표본점이 차지한 복셀 수 x 복셀 부피 (복셀 한 변 = 2 x 격자 dx)
  N_min 의 세 판 (a, b, c) 평균 (없는 판은 건너뜀).
시각 (같은 렌더러 PNG, 프레임 1..T):
  무참조 품질 NIQE↓, BRISQUE↓, CLIP-IQA↑, MUSIQ↑ (pyiqa 기본 가중치), 프레임 평균
  깜빡임 warping error↓ = 프레임 t+1 을 RAFT (torchvision raft_large) 광류 F_{t->t+1} 로 t 에 맞춘 뒤 mean_{가려지지 않은 픽셀} ||I_t - W(I_{t+1})||^2 (RGB [0,1]),
     가려짐 = 앞-뒤 광류 일관성 |F_f + F_b(warp)|^2 > 0.01 (|F_f|^2 + |F_b|^2) + 0.5, 프레임 쌍 평균
  기준 대비 PSNR·SSIM·LPIPS 는 vis.json (stab_vis.py) 그대로.
  python exe/stab_extra.py --scene mic_metal_lego --hextag _f5 [--no_visual]
"""
import argparse, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import stab1115_lib as L  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--hextag", default="_f5")
ap.add_argument("--no_visual", action="store_true"); ap.add_argument("--out", default="")
a = ap.parse_args()
S = a.scene; ST = L.ST; D = f"{ST}/{S}"; V = f"{D}/vis"
cfg, zf, g, dt, dxg = L.scene_consts(S); FR = L.nframes(S)
OUT = a.out or f"{D}/extra.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def items():
    it = [("pg_N1024", "pg", 1024)]
    for m in ("pg", "ipg", "omni", "ours"):
        p = f"{D}/plan_{m}{a.hextag if m == 'ours' else ''}.json"
        if os.path.exists(p):
            n = json.load(open(p)).get("N_min")
            if n:
                it.append((f"{m}{a.hextag if m == 'ours' else ''}_N{n}", m, int(n)))
    return it


def load(m, n, rep):
    if m == "ours":
        return L.hex_load(L.hex_dir(S, n, rep) + a.hextag, FR)
    r = L.load(S, m, n, rep)
    if r is None and rep == "a" and os.path.exists(f"{V}/{m}_N{n}.npz"):       # 레고 PG·i-PG (11-12 기존 결과) = vis 단계에서 다시 돌린 판
        import stab_metrics as sm
        r = sm.load(V, m, n, FR)
    return r


def phys(r):
    o = {}
    rows, X = r.get("rows"), r.get("X")
    if rows is not None and "ke" in rows and "pe" in rows:
        E = np.asarray(rows["ke"], np.float64) + np.asarray(rows["pe"], np.float64)
        o["energy_inc"] = float(max(0.0, np.nanmax((E - E[0]) / abs(E[0]))))
    if X is not None and len(X) > 1:
        h = 2 * dxg; vs = []
        for t in range(len(X)):
            x = X[t]
            if not np.isfinite(x).all():
                break
            vs.append(len(np.unique(np.floor(x / h).astype(np.int64), axis=0)))
        vs = np.array(vs, np.float64)
        o["vol_change"] = float(np.max(np.abs(vs / vs[0] - 1.0)))
        o["vol_change_last"] = float(abs(vs[-1] / vs[0] - 1.0))
    return o


PH = {}
for name, m, n in items():
    runs = []
    for rep in (("a",) if name == "pg_N1024" else L.REPS):
        r = load(m, n, rep)
        if r is not None and r.get("status") == "끝":
            runs.append(phys(r))
    if runs:
        PH[name] = {k: float(np.mean([q[k] for q in runs if k in q])) for k in runs[0]} | {"runs": len(runs)}
    print("[물리]", name, PH.get(name), flush=True)
res["phys"] = PH
json.dump(res, open(OUT, "w"), indent=1)

if not a.no_visual:
    import imageio.v2 as imageio
    import torch
    import torch.nn.functional as Fn
    import pyiqa
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    dev = "cuda"
    IQ = {k: pyiqa.create_metric(k, device=dev) for k in ("niqe", "brisque", "clipiqa", "musiq")}
    RAFT = raft_large(weights=Raft_Large_Weights.DEFAULT).to(dev).eval()

    def img(p):
        return torch.as_tensor(imageio.imread(p)[..., :3].astype(np.float32) / 255).permute(2, 0, 1)[None].to(dev)

    def warp(x, f):
        B, _, H, W = x.shape
        gy, gx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
        gx = (gx[None] + f[:, 0]) / (W - 1) * 2 - 1; gy = (gy[None] + f[:, 1]) / (H - 1) * 2 - 1
        return Fn.grid_sample(x, torch.stack([gx, gy], -1), align_corners=True)

    def flow(a_, b_):
        H, W = a_.shape[-2:]; H8, W8 = (H + 7) // 8 * 8, (W + 7) // 8 * 8
        pa = Fn.pad(a_, (0, W8 - W, 0, H8 - H), mode="replicate") * 2 - 1; pb = Fn.pad(b_, (0, W8 - W, 0, H8 - H), mode="replicate") * 2 - 1
        return RAFT(pa, pb)[-1][..., :H, :W]

    VIS = {}
    for name, m, n in items():
        P = sorted(glob.glob(f"{V}/png_{name}/*.png"))
        if len(P) < 3:
            print("[시각] PNG 없음", name, flush=True); continue
        P = P[1:]                                                     # 0 프레임 = 모두 같은 처음 상태
        q = {k: [] for k in IQ}; we = []
        with torch.no_grad():
            prev = None
            for p in P:
                x = img(p)
                for k, f in IQ.items():
                    q[k].append(float(f(x)))
                if prev is not None:
                    ff, fb = flow(prev, x), flow(x, prev)
                    fbw = warp(fb, ff)
                    occ = ((ff + fbw) ** 2).sum(1) > 0.01 * ((ff ** 2).sum(1) + (fbw ** 2).sum(1)) + 0.5
                    msk = (~occ).float()[:, None]
                    d = ((prev - warp(x, ff)) ** 2).sum(1, keepdim=True)
                    we.append(float((d * msk).sum() / msk.sum().clamp(min=1)))
                prev = x
        VIS[name] = {k: float(np.mean(v)) for k, v in q.items()} | dict(warp_err=float(np.mean(we)), frames=len(P))
        print("[시각]", name, VIS[name], flush=True)
    res["visual"] = VIS
    json.dump(res, open(OUT, "w"), indent=1)
print("[stab_extra] ->", OUT)
