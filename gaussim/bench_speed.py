"""GausSim 공식 코드의 시뮬레이션 프레임당 비용.

렌더링은 제외한다 -- 우리 쪽 비교 수치(MPM, 학생 스테퍼, Simplicits 축소)도 전부
렌더링을 빼고 잰 값이라 같은 잣대로 맞추기 위해서다. 그래서 래스터라이저는
스텁으로 대체하고, decode_head.pre_render 도 no-op 으로 바꾼다.

가중치는 무작위 초기화다. 속도는 가중치 값과 무관하고, 공개된 데모에 체크포인트가
포함되어 있지 않다.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import types

import numpy as np


def stub_render_deps(gs_path):
    """래스터라이저·KNN 을 스텁으로 채운다 (속도 측정에 렌더링은 포함하지 않는다)."""
    m = types.ModuleType("diff_gaussian_rasterization")

    class GaussianRasterizationSettings:
        def __init__(self, *a, **k):
            pass

    class GaussianRasterizer:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            raise RuntimeError("렌더링은 이 벤치마크에 포함되지 않는다")

    m.GaussianRasterizationSettings = GaussianRasterizationSettings
    m.GaussianRasterizer = GaussianRasterizer
    sys.modules["diff_gaussian_rasterization"] = m
    k = types.ModuleType("simple_knn")
    kc = types.ModuleType("simple_knn._C")
    kc.distCUDA2 = lambda x: x.new_zeros(x.shape[0])
    k._C = kc
    sys.modules["simple_knn"] = k
    sys.modules["simple_knn._C"] = kc
    sys.path.insert(0, gs_path)


ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--scene", default="pudding")
ap.add_argument("--data_dir", default="data_real")
ap.add_argument("--gs_path", required=True, help="scene.gaussian_model 이 있는 경로")
ap.add_argument("--ply", default=None,
                help="다른 장면의 ply. 주면 그 장면을 GausSim 방식으로 계층화해 잰다")
ap.add_argument("--n_mov", type=int, default=8926,
                help="이동 가우시안 수 (pudding 과 같은 규모로 맞추기 위한 기본값)")
ap.add_argument("--lv1", type=int, default=961, help="1 단계 클러스터 목표 개수")
ap.add_argument("--lv2", type=int, default=8, help="2 단계 클러스터 목표 개수")
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()

stub_render_deps(a.gs_path)
import mmcv
import torch

from mmgs.models import build_simulator

dev = "cuda"
cfg = mmcv.Config.fromfile(a.config)
cfg.model.pretrained = None
cfg.model.force_forward = -1
print(f"[설정] {a.config}")
print(f"  계층 {cfg.model.get('cluster_cfg')}")
print(f"  dt {cfg.model.get('dt')}  백본층 {cfg.model.backbone['num_encoder_layers']} "
      f"embed {cfg.model.backbone['embed_dims']}")

# 이 클래스의 train() 은 self 를 돌려주지 않아 .eval() 이 None 이 된다. 따로 부른다.
model = build_simulator(cfg.model)
model.to(dev)
model.eval()
model.decode_head.pre_render = lambda *x, **k: (torch.zeros(3, 4, 4, device=dev),)
SC = a.scene
gaussian = model.gs_scene_dict[SC]
print(f"[장면] {SC}: 가우시안 {gaussian.get_xyz.shape[0]}")


def cluster_like_gaussim(pts, target, lo, hi):
    """공식 코드와 같은 방식(complete-linkage 응집형 + NearestCentroid).

    거리 임계값만 장면 규모에 맞게 이분 탐색한다 -- 그들도 장면마다 따로 정한다
    (pudding 은 0.04, 0.4).
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import NearestCentroid
    best = None
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        lab = AgglomerativeClustering(n_clusters=None, linkage="complete",
                                      distance_threshold=mid).fit(pts).labels_
        k = int(lab.max()) + 1
        if best is None or abs(k - target) < abs(best[1] - target):
            best = (lab, k, mid)
        if k > target:
            lo = mid
        else:
            hi = mid
        if k == target:
            break
    lab = best[0]
    cen = NearestCentroid().fit(pts, lab).centroids_
    return lab, cen, best[1], best[2]


if a.ply is not None:
    import torch as _t
    from scene.gaussian_model import GaussianModel as _GM
    g2 = _GM(3)
    g2.load_ply(a.ply)
    xyz_all = g2.get_xyz.detach()
    op = _t.sigmoid(g2._opacity).reshape(-1) if hasattr(g2, "_opacity") else None
    keep = (op > 0.02) if op is not None else _t.ones(xyz_all.shape[0], dtype=_t.bool,
                                                      device=xyz_all.device)
    idx_all = _t.nonzero(keep).reshape(-1)
    rs = np.random.RandomState(0)
    sel = idx_all[_t.as_tensor(np.sort(rs.choice(idx_all.numel(),
                                                 min(a.n_mov, idx_all.numel()),
                                                 replace=False))).to(idx_all.device)]
    N = int(sel.numel())
    xyz = xyz_all[sel].contiguous()
    scal = g2.get_scaling.detach()[sel].contiguous()
    cov = g2.get_covariance().detach()[sel].contiguous()
    ext = float((xyz.max(0).values - xyz.min(0).values).norm())
    pts = xyz.cpu().numpy()
    l1, c1, k1, d1 = cluster_like_gaussim(pts, a.lv1, 1e-4 * ext, 0.5 * ext)
    l2, c2, k2, d2 = cluster_like_gaussim(c1, a.lv2, 1e-3 * ext, 1.0 * ext)
    # 고정점: 물체 아래쪽 5% (화분). 공식 파이프라인과 같이 마지막 단계는
    # 최상위 클러스터 -> 가장 가까운 고정 정점 매핑이다.
    zaxis = pts[:, 2]
    thr = np.quantile(zaxis, 0.05)
    pin_idx = np.nonzero(zaxis <= thr)[0]
    from sklearn.metrics.pairwise import euclidean_distances
    # 공식 코드는 마지막 매핑의 max+1 을 고정 정점 수로 삼는다
    # (_preprocess_abs_hierarchy: n_clusters = max(connect_edges)+1).
    # 그래서 실제로 쓰이는 고정 정점만 남기고 조밀하게 다시 번호를 매긴다.
    _near = np.argmin(euclidean_distances(c2, pts[pin_idx]), axis=-1)
    _used, l3 = np.unique(_near, return_inverse=True)
    pin_idx = pin_idx[_used]
    l3 = l3.astype(np.int64)
    p2c = [torch.as_tensor(x).long().to(dev) for x in (l1, l2, l3)]
    print(f"[장면] {os.path.basename(a.ply)}: 가우시안 {int(idx_all.numel())} 중 "
          f"이동 {N}, 크기 {ext:.3f}")
    print(f"[계층] {N} -> {k1} (임계 {d1:.4f}) -> {k2} (임계 {d2:.4f}) -> 1, "
          f"고정점 {len(pin_idx)}")
    pin_mask = torch.zeros(N, 1, device=dev)
    pin_mask[torch.as_tensor(pin_idx).long().to(dev)] = 1.0
    vs = 300.0
    diag_volume = torch.prod(scal * vs, dim=-1, keepdim=True)
    attr = torch.zeros(N, model.attr_dim, device=dev)
    ext_f = torch.zeros(N, 3, device=dev)
    gaussian = g2
    mov_mask = torch.zeros(int(xyz_all.shape[0]), dtype=torch.bool, device=dev)
    mov_mask[sel] = True

    def one_frame(prev, cur, timing):
        torch.cuda.synchronize(); t0 = time.time()
        ig, cg = model._preprocess(prev, cur, xyz, attr, diag_volume, cov, ext_f,
                                   p2c_mapping=p2c, pin_mask=pin_mask)
        torch.cuda.synchronize(); t1 = time.time()
        pred_pos, _, _, _, _ = model.encode_decode(
            ig, cg, gaussian, [None], gaussian.get_covariance().clone(),
            gaussian.get_xyz.clone(), mov_mask, gt_label=None)
        torch.cuda.synchronize(); t2 = time.time()
        timing[0] += t1 - t0
        timing[1] += t2 - t1
        return cur, pred_pos.detach()

    with torch.no_grad():
        prev, cur = xyz.clone(), xyz.clone()
        for _ in range(3):
            prev, cur = one_frame(prev, cur, [0.0, 0.0])
        best = None
        for r in range(a.reps):
            prev, cur = xyz.clone(), xyz.clone()
            tm = [0.0, 0.0]
            for _ in range(a.frames):
                prev, cur = one_frame(prev, cur, tm)
            tot = sum(tm) / a.frames * 1000
            if best is None or tot < best[0]:
                best = (tot, tm[0] / a.frames * 1000, tm[1] / a.frames * 1000)
    print(f"\n[결과] 프레임당 {best[0]:.2f} ms "
          f"(그래프 생성 {best[1]:.2f} + 신경망 {best[2]:.2f}), 렌더링 제외")
    print(f"[결과] 이동 가우시안 {N}, 계층 {N} -> {k1} -> {k2} -> 1")
    sys.exit(0)

D = os.path.join(a.data_dir, SC)
mov_mask = torch.as_tensor(pickle.load(open(os.path.join(D, "pc_mask.pkl"), "rb"))).to(dev)
if mov_mask.dtype != torch.bool:
    mov_mask = mov_mask.bool()
mov_mask = mov_mask.reshape(-1)
N = int(mov_mask.sum())
print(f"[장면] 이동 가우시안 {N}")

cl = [f for f in os.listdir(D) if f.endswith("cluster_mask.pkl")]
assert cl, f"{D} 에 cluster_mask.pkl 이 없다"
p2c_raw = pickle.load(open(os.path.join(D, cl[0]), "rb"))["p2c"]
p2c = [torch.as_tensor(np.asarray(x)).long().to(dev) for x in p2c_raw]
print(f"[계층] {cl[0]}  ->  " + " -> ".join(
    [str(N)] + [str(int(x.max()) + 1) for x in p2c]))

pin = json.load(open(os.path.join(D, "pin_mask.json")))
pin_mask = torch.zeros(N, 1, device=dev)
try:
    idx = torch.as_tensor(np.asarray(pin[0])).reshape(-1).long().to(dev)
    if idx.numel() and idx.max() < N:
        pin_mask[idx] = 1.0
except Exception as e:
    print(f"[경고] pin_mask 해석 실패 ({e}) -- 0 으로 둔다")
print(f"[장면] 고정점 {int(pin_mask.sum())}/{N}")

xyz = gaussian.get_xyz[mov_mask]
scal = gaussian.get_scaling[mov_mask]
cov = gaussian.get_covariance()[mov_mask]
vs = cfg.get("volume_scalar", {}).get(SC, 1.0)
diag_volume = torch.prod(scal * vs, dim=-1, keepdim=True)
attr = model.scene_attr[SC]
ext = torch.zeros(N, 3, device=dev)
print(f"[장면] volume_scalar {vs}, attr {tuple(attr.shape)}, cov {tuple(cov.shape)}")


def one_frame(prev, cur, timing):
    torch.cuda.synchronize(); t0 = time.time()
    ig, cg = model._preprocess(prev, cur, xyz, attr, diag_volume, cov, ext,
                               p2c_mapping=p2c, pin_mask=pin_mask)
    torch.cuda.synchronize(); t1 = time.time()
    pred_pos, pred_cov, _, _, _ = model.encode_decode(
        ig, cg, gaussian, [None], gaussian.get_covariance().clone(),
        gaussian.get_xyz.clone(), mov_mask, gt_label=None)
    torch.cuda.synchronize(); t2 = time.time()
    timing[0] += t1 - t0
    timing[1] += t2 - t1
    return cur, pred_pos.detach()


with torch.no_grad():
    prev, cur = xyz.clone(), xyz.clone()
    for _ in range(3):                         # 예열
        prev, cur = one_frame(prev, cur, [0.0, 0.0])
    best = None
    for r in range(a.reps):
        prev, cur = xyz.clone(), xyz.clone()
        tm = [0.0, 0.0]
        for _ in range(a.frames):
            prev, cur = one_frame(prev, cur, tm)
        tot = sum(tm) / a.frames * 1000
        if best is None or tot < best[0]:
            best = (tot, tm[0] / a.frames * 1000, tm[1] / a.frames * 1000)
    print(f"\n[결과] 프레임당 {best[0]:.2f} ms "
          f"(그래프 생성 {best[1]:.2f} + 신경망 {best[2]:.2f}), 렌더링 제외")
    print(f"[결과] 이동 가우시안 {N}, 계층 " + " -> ".join(
        [str(N)] + [str(int(x.max()) + 1) for x in p2c]))
