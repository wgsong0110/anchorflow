"""3DGS 껍데기의 속을 **포아송 디스크 입자셋**으로 채운다.

PhysGaussian 의 `particle_filling` 은 레이캐스팅 + 밀도 문턱이라 칸당 한 점이고
분포가 표면 밀도를 따라간다. 여기서는 점유 격자의 속을 메우고 그 안에 최소 간격을
지키는 무작위 점을 뿌린다 -- 격자로 채우면 줄무늬가 렌더에 그대로 남는다.

3DGS 껍데기에는 구멍이 뚫려 있어서 `binary_fill_holes` 만 쓰면 바깥에서 새어 들어와
속이 안 메워진다. 그래서 먼저 닫음(dilate->fill->erode)으로 틈을 막고, 세 축 단면별
2D 채움 중 둘 이상에서 안쪽인 칸을 내부로 본다.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (binary_dilation, binary_erosion, binary_fill_holes,
                           gaussian_filter)
from scipy.spatial import cKDTree
from tqdm import tqdm


def solid_mask(S, grid=160, level=0.5, close=0.03, pad=0.04):
    """표면 점 S 로부터 내부(속이 메워진) 점유 격자를 만든다."""
    N = grid
    lo, hi = S.min(0) - pad, S.max(0) + pad
    h = (hi - lo) / N
    gi = np.floor((S - lo) / h).astype(int).clip(0, N - 1)
    vol = np.zeros((N, N, N), np.float32)
    np.add.at(vol, (gi[:, 0], gi[:, 1], gi[:, 2]), 1.0)
    vol = gaussian_filter(vol, sigma=1.0)
    thr = float(np.quantile(vol[vol > 1e-6], level))
    occ = vol > thr

    kc = max(1, int(round(close / float(h.min()))))
    occ_c = binary_dilation(occ, iterations=kc)

    def _fill2d(m, ax):
        out = np.zeros_like(m)
        for i in range(m.shape[ax]):
            sl = [slice(None)] * 3
            sl[ax] = i
            out[tuple(sl)] = binary_fill_holes(m[tuple(sl)])
        return out

    f = sum(_fill2d(occ_c, ax).astype(np.uint8) for ax in (0, 1, 2))
    solid = binary_fill_holes(occ_c) | (f >= 2)
    solid = binary_erosion(solid, iterations=kc + 1)
    return solid, lo, hi, h, occ, kc


def poisson_fill(S, spacing=0.012, grid=160, level=0.5, close=0.03,
                 surf=0.3, seed=0, verbose=True):
    """S(표면 점 Nx3) 안쪽을 간격 `spacing` 의 입자셋으로 채워 돌려준다."""
    S = np.asarray(S, np.float64)
    solid, lo, hi, h, occ, kc = solid_mask(S, grid, level, close)
    N = grid
    cell_v = float(h[0] * h[1] * h[2])
    target = int(int(solid.sum()) * cell_v / (spacing ** 3))
    if verbose:
        print(f"[격자] {N}^3 점유 {int(occ.sum())} -> 속 채움 {int(solid.sum())} 칸 "
              f"(닫음 {kc}칸), 목표 입자 {target}", flush=True)

    rng = np.random.default_rng(seed)
    need = max(target * 60, 200000)
    chunks, got = [], 0
    while got < need:
        c = rng.uniform(lo, hi, size=(2_000_000, 3))
        ci = np.floor((c - lo) / h).astype(int).clip(0, N - 1)
        c = c[solid[ci[:, 0], ci[:, 1], ci[:, 2]]]
        if c.shape[0] == 0:
            break
        chunks.append(c)
        got += c.shape[0]
    cand = np.concatenate(chunks, 0)[:need]

    r_min = 0.85 * spacing
    order = rng.permutation(len(cand))
    gk = np.floor(cand / r_min).astype(int)
    grid_key, acc = {}, []
    offs = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    it = tqdm(order, desc="포아송 디스크", ncols=80) if verbose else order
    for i in it:
        k0, k1, k2 = gk[i]
        nb = []
        for dx_, dy_, dz_ in offs:
            v = grid_key.get((k0 + dx_, k1 + dy_, k2 + dz_))
            if v:
                nb.extend(v)
        if nb and (np.linalg.norm(cand[nb] - cand[i], axis=1) < r_min).any():
            continue
        grid_key.setdefault((k0, k1, k2), []).append(i)
        acc.append(i)
    L = cand[np.array(acc)]

    n0 = L.shape[0]
    d, nn = cKDTree(S).query(L, k=1)
    L = L[d > surf * spacing]                      # 표면 커널과 겹치지 않게
    if verbose:
        print(f"[채움] 포아송 {n0} -> 표면필터 후 {L.shape[0]} 개 "
              f"(최소 간격 {r_min:.4f})", flush=True)
    return L


def mesh_solid(S, grid=160, sigma=2.0, level=0.5, pad=0.04, mesh_out=None):
    """3DGS -> **메시** -> 메시 내부(복셀) 를 돌려준다.

    exe/gs_to_mesh.py 와 같은 방식이다: 가우시안 중심을 격자에 담고 번지게 한 뒤
    marching cubes 로 등위면을 뽑는다. 그 등위면이 곧 vol == thr 이므로 메시 내부는
    vol > thr 이고, 갇힌 빈 공간만 binary_fill_holes 로 메운다.
    """
    import mcubes
    N = grid
    lo, hi = S.min(0) - pad, S.max(0) + pad
    h = (hi - lo) / N
    gi = np.floor((S - lo) / h).astype(int).clip(0, N - 1)
    vol = np.zeros((N, N, N), np.float32)
    np.add.at(vol, (gi[:, 0], gi[:, 1], gi[:, 2]), 1.0)
    vol = gaussian_filter(vol, sigma=sigma)
    thr = float(np.quantile(vol[vol > 1e-6], level))
    v, f = mcubes.marching_cubes(vol, thr)
    if mesh_out:
        mcubes.export_obj(v * h + lo, f, mesh_out)
    solid = binary_fill_holes(vol > thr)
    solid = binary_erosion(solid, iterations=1)
    print(f"[메시] {N}^3 번짐 {sigma} 문턱 {thr:.4f} -> 꼭짓점 {v.shape[0]} "
          f"면 {f.shape[0]}, 내부 {int(solid.sum())} 칸", flush=True)
    return solid, lo, hi, h


def poisson_fill_mesh(S, spacing=0.012, grid=160, sigma=2.0, level=0.5,
                      surf=0.3, seed=0, mesh_out=None, verbose=True):
    """3DGS 에서 뽑은 **메시 내부**를 포아송 디스크 입자셋으로 채운다."""
    S = np.asarray(S, np.float64)
    solid, lo, hi, h = mesh_solid(S, grid, sigma, level, mesh_out=mesh_out)
    N = grid
    cell_v = float(h[0] * h[1] * h[2])
    target = int(int(solid.sum()) * cell_v / (spacing ** 3))
    rng = np.random.default_rng(seed)
    need = max(target * 60, 200000)
    chunks, got = [], 0
    while got < need:
        c = rng.uniform(lo, hi, size=(2_000_000, 3))
        ci = np.floor((c - lo) / h).astype(int).clip(0, N - 1)
        c = c[solid[ci[:, 0], ci[:, 1], ci[:, 2]]]
        if c.shape[0] == 0:
            break
        chunks.append(c)
        got += c.shape[0]
    cand = np.concatenate(chunks, 0)[:need]

    r_min = 0.85 * spacing
    order = rng.permutation(len(cand))
    gk = np.floor(cand / r_min).astype(int)
    grid_key, acc = {}, []
    offs = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    it = tqdm(order, desc="포아송 디스크", ncols=80) if verbose else order
    for i in it:
        k0, k1, k2 = gk[i]
        nb = []
        for dx_, dy_, dz_ in offs:
            v = grid_key.get((k0 + dx_, k1 + dy_, k2 + dz_))
            if v:
                nb.extend(v)
        if nb and (np.linalg.norm(cand[nb] - cand[i], axis=1) < r_min).any():
            continue
        grid_key.setdefault((k0, k1, k2), []).append(i)
        acc.append(i)
    L = cand[np.array(acc)]
    n0 = L.shape[0]
    d, _ = cKDTree(S).query(L, k=1)
    L = L[d > surf * spacing]
    if verbose:
        print(f"[채움] 목표 {target}, 포아송 {n0} -> 표면필터 후 {L.shape[0]} 개",
              flush=True)
    return L
