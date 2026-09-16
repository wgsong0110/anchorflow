"""가우시안 하나가 실제로 몇 개의 앵커에 매여 있는지 분포를 그린다.

잘린 가우시안은 지지가 G(x) > c 인 영역이라 **가우시안마다 붙는 앵커 수가 다르다**
-- 이것이 kNN-softmax(항상 정확히 k 개)와 갈리는 지점이다. 그 수가 실제로 어떻게
퍼져 있는지, 그리고 기하 학습이 그것을 어느 쪽으로 미는지 본다.

두 가지를 구별해서 센다.

  후보 짝   refresh 가 KD-트리로 잡아온 개수. 마할라노비스 반경에 여유(margin)를
            두고 뽑은 **상위집합**이라 실제 지지보다 크다.
  유효 지지 정규화된 가중치가 문턱을 넘는 앵커 수. 이쪽이 "영향을 미치는 앵커 수"다.
            잘린 가우시안은 경계 밖이 정확히 0 이므로 문턱 0 이 곧 정의이고,
            softmax 는 어디서도 0 이 아니라 문턱을 어디 두느냐로 값이 바뀐다 --
            그래서 여러 문턱에서 같이 찍는다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--h5_dir", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--geom", default=None,
                help="학습된 기하 .pt. 주면 학습 전/후를 함께 그린다")
ap.add_argument("--geom_softmax", default=None,
                help="비교용 softmax 기하 .pt")
ap.add_argument("--softmax_k", type=int, default=16)
ap.add_argument("--n_anchors", type=int, default=512)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--n_pts", type=int, default=40000)
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--c", type=float, default=0.25)
ap.add_argument("--eig_floor", type=float, default=0.02)
ap.add_argument("--thresh", type=float, nargs="+", default=[0.0, 0.01, 0.05],
                help="정규화 가중치가 이보다 커야 '영향을 미친다'로 센다")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda"
torch.set_grad_enabled(False)

from anchorflow import gf_scene                   # noqa: E402
from anchorflow.anchor_sparse import AnchorSparse  # noqa: E402

cfg = gf_scene.read_cfg(a.config)
TR, EXT = gf_scene.load_traj(a.h5_dir, stride=a.stride, n_pts=a.n_pts,
                             seed=a.seed, dev=dev)
X0 = TR[0].contiguous()
sc = gf_scene.build_scene(X0, cfg, n_anchors=a.n_anchors, K=a.K,
                          eig_floor=a.eig_floor, dev=dev)
del TR


def support(softmax_w, geom):
    fit = AnchorSparse(sc, c=a.c, eig_floor=a.eig_floor, softmax_w=softmax_w,
                       softmax_k=a.softmax_k).to(dev)
    if geom:
        st = torch.load(geom, map_location=dev, weights_only=False)
        for k in ("pos", "log_s", "quat", "log_amp"):
            getattr(fit, k).data.copy_(st[k].to(dev))
    fit.refresh()
    fit.set_B_ref()
    w = fit.weights()
    N = fit.N
    cand = torch.zeros(N, device=dev).index_add_(
        0, fit.pair_g, torch.ones_like(w))
    out = {"cand": cand.cpu().numpy()}
    for t in a.thresh:
        cnt = torch.zeros(N, device=dev).index_add_(
            0, fit.pair_g, (w > t).float())
        out[f"eff{t:g}"] = cnt.cpu().numpy()
    # 유효 자유도: 1/sum(w^2) -- 문턱에 기대지 않는 "몇 개가 실질적으로 나누는가"
    w2 = torch.zeros(N, device=dev).index_add_(0, fit.pair_g, w * w)
    out["ess"] = (1.0 / w2.clamp(min=1e-20)).cpu().numpy()
    del fit
    torch.cuda.empty_cache()
    return out


CASES = [("잘린 가우시안 · 학습 전", False, None)]
if a.geom:
    CASES.append(("잘린 가우시안 · 학습 후", False, a.geom))
CASES.append(("kNN softmax k=%d · 학습 전" % a.softmax_k, True, None))
if a.geom_softmax:
    CASES.append(("kNN softmax k=%d · 학습 후" % a.softmax_k, True, a.geom_softmax))

res = {}
for name, sm, geom in CASES:
    r = support(sm, geom)
    res[name] = r
    e0 = r[f"eff{a.thresh[0]:g}"]
    print(f"[{name}]  후보 {r['cand'].mean():.1f}  "
          f"유효(>{a.thresh[0]:g}) 평균 {e0.mean():.2f} 중앙 {np.median(e0):.0f} "
          f"범위 {e0.min():.0f}~{e0.max():.0f}  ESS {r['ess'].mean():.2f}",
          flush=True)

# ---------------------------------------------------------------- 그림
import matplotlib                                   # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

os.makedirs(a.out, exist_ok=True)
key = f"eff{a.thresh[0]:g}"
hi = int(max(res[n][key].max() for n in res))
bins = np.arange(-0.5, hi + 1.5)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
for name in res:
    ax[0].hist(res[name][key], bins=bins, histtype="step", lw=1.8,
               density=True, label=name)
    ax[1].hist(res[name]["ess"], bins=60, histtype="step", lw=1.8,
               density=True, label=name)
ax[0].set_xlabel(f"가우시안당 영향 앵커 수 (가중치 > {a.thresh[0]:g})")
ax[0].set_ylabel("비율")
ax[0].set_title("유효 지지 크기")
ax[1].set_xlabel("유효 자유도 1 / Σw²")
ax[1].set_title("가중치가 실제로 몇 개로 쪼개지나")
for x in ax:
    x.legend(fontsize=8)
    x.grid(alpha=0.3)
fig.suptitle(f"{cfg.get('material')} · 앵커 {sc.M} · 입자 {X0.shape[0]}")
fig.tight_layout()
p_png = os.path.join(a.out, "anchor_support.png")
fig.savefig(p_png, dpi=140)
print(f"[저장] {p_png}", flush=True)

summ = {}
for name, r in res.items():
    d = {"cand_mean": float(r["cand"].mean()), "ess_mean": float(r["ess"].mean())}
    for t in a.thresh:
        e = r[f"eff{t:g}"]
        d[f"eff{t:g}"] = dict(mean=float(e.mean()), median=float(np.median(e)),
                              p5=float(np.percentile(e, 5)),
                              p95=float(np.percentile(e, 95)),
                              min=int(e.min()), max=int(e.max()),
                              hist=np.bincount(e.astype(int)).tolist())
    summ[name] = d
json.dump(dict(n_anchors=int(sc.M), n_pts=int(X0.shape[0]),
               thresh=a.thresh, cases=summ),
          open(os.path.join(a.out, "anchor_support.json"), "w"), indent=1,
          ensure_ascii=False)
print("SUPPORT_OK")
