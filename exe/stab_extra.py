"""[stabfinal] 안정성 추가 지표 (한 장면, 확정 최소 N 의 모든 방법 + 기준 PG N=1024) -> <장면>/extra.json
물리 (정답 없음, 궤적 표본점 X 와 에너지 기록 rows):
  energy_inc = max(0, max_t (E_t - E_0) / |E_0|), E = 운동 + 위치 에너지 (안정 판정과 같은 E; 탄성 에너지는 방법마다 기록 여부가 달라 뺌)
  e_drift_max·e_ratio_last·jdev_max·detneg_max = R2 lbours/final_fix/metrics_def.md 와 같은 키·정의 (프레임 통계 stabstat.frame_stats 의 jdev = mean|J-1|)
  vol_change (보조) = max_t |V_t / V_0 - 1|, V_t = 표본점이 차지한 복셀 수 x 복셀 부피 (복셀 한 변 = 2 x 격자 dx)
  N_min 의 세 판 (a, b, c) 평균 (없는 판은 건너뜀).
시각 (같은 렌더러 PNG, 프레임 1..T):
  무참조 품질 NIQE↓, BRISQUE↓, CLIP-IQA↑, MUSIQ↑ (pyiqa 기본 가중치), 프레임 평균
  깜빡임 warping error↓ = metrics_def.md (final_fix exe/ffx_vis.py 와 같은 코드): I_t 를 뒤 광류 F_b (RAFT C_T_SKHT_V2) 로 끌어와 I_{t+1} 과의 차 제곱 (RGB 채널 합),
     마스크 = 앞뒤 일관성 |F_b + F_f(x+F_b)|^2 < 0.01 (|F_b|^2 + |F_f|^2) + 0.5 & 영상 안, 프레임 0->1 .. T-1->T 평균
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
        o["e_drift_max"] = float(np.nanmax((E - E[0]) / abs(E[0])))              # R2 lbours/final_fix/metrics_def.md 키 (부호 그대로)
        o["e_ratio_last"] = float(E[-1] / E[0])
    if rows is not None and "jdev" in rows and np.isfinite(rows["jdev"]).any():     # metrics_def jdev_max: 프레임마다 mean |J-1| (J = det F), 그 최대
        o["jdev_max"] = float(np.nanmax(rows["jdev"]))
    if rows is not None and "detneg" in rows and np.isfinite(rows["detneg"]).any():
        o["detneg_max"] = float(np.nanmax(rows["detneg"]))
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
        PH[name] = {k: float(np.mean([q[k] for q in runs if k in q])) for k in sorted({k for q in runs for k in q})} | {"runs": len(runs)}
    print("[물리]", name, PH.get(name), flush=True)
res["phys"] = PH
json.dump(res, open(OUT, "w"), indent=1)

if os.environ.get("SX_PHYS_ONLY") or a.no_visual:
    pass
else:
    # [CM191] 시각 지표는 R2 lbours/final_fix/metrics_def.md · exe/ffx_vis.py (final_fix) 와 같은 코드: IQA = 프레임 1..T 평균 (clamp 0~1),
    #   warp_err = 프레임 0->1 .. T-1->T, I_t 를 뒤 광류 F_b (t+1 -> t) 로 끌어와 I_{t+1} 과 비교, 마스크 = 앞뒤 일관성 & 영상 안, RAFT C_T_SKHT_V2 + 그 transforms
    import imageio.v2 as imageio
    import torch
    import torch.nn.functional as Fn
    import pyiqa
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    dev = "cuda"
    IQ = {k: pyiqa.create_metric(k, device=dev) for k in ("niqe", "brisque", "clipiqa", "musiq")}
    W8 = Raft_Large_Weights.C_T_SKHT_V2
    RAFT = raft_large(weights=W8, progress=False).to(dev).eval(); TF = W8.transforms()

    def img(p):
        return torch.as_tensor(imageio.imread(p)[..., :3].astype(np.float32) / 255).permute(2, 0, 1)[None].to(dev)

    def pad8(x):
        H, W = x.shape[-2:]
        return Fn.pad(x, (0, (8 - W % 8) % 8, 0, (8 - H % 8) % 8), mode="replicate"), H, W

    def flow(i1, i2):
        a_, b_ = TF(i1, i2)
        a_, H, W = pad8(a_); b_, _, _ = pad8(b_)
        return RAFT(a_, b_)[-1][..., :H, :W]

    def warp(x, f):
        _, _, H, W = x.shape
        yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
        gx = xx[None] + f[:, 0]; gy = yy[None] + f[:, 1]
        inside = (gx >= 0) & (gx <= W - 1) & (gy >= 0) & (gy <= H - 1)
        grid = torch.stack([2 * gx / (W - 1) - 1, 2 * gy / (H - 1) - 1], -1)
        return Fn.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True), inside

    VIS = {}
    for name, m, n in items():
        P = sorted(glob.glob(f"{V}/png_{name}/*.png"))
        if len(P) < 3:
            print("[시각] PNG 없음", name, flush=True); continue
        q = {k: [] for k in IQ}; we, oc = [], []
        with torch.no_grad():
            prev = img(P[0])
            for p in P[1:]:
                cur = img(p)
                for k, f in IQ.items():
                    q[k].append(float(f(cur.clamp(0, 1))))
                fb = flow(cur, prev); ff = flow(prev, cur)
                wprev, ins = warp(prev, fb); ffw, _ = warp(ff, fb)
                Mk = (((fb + ffw) ** 2).sum(1) < 0.01 * ((fb ** 2).sum(1) + (ffw ** 2).sum(1)) + 0.5) & ins
                e = ((cur - wprev) ** 2).sum(1)
                we.append(float((e * Mk).sum() / Mk.sum().clamp_min(1))); oc.append(float(1 - Mk.float().mean()))
                prev = cur
        VIS[name] = {k: float(np.mean(v)) for k, v in q.items()} | dict(warp_err=float(np.mean(we)), warp_occ_frac=float(np.mean(oc)), frames=len(P) - 1,
                                                                        pyiqa_version=getattr(pyiqa, "__version__", "?"), def_="metrics_def.md (final_fix)")
        print("[시각]", name, VIS[name], flush=True)
    res["visual"] = VIS
    json.dump(res, open(OUT, "w"), indent=1)
print("[stab_extra] ->", OUT)
