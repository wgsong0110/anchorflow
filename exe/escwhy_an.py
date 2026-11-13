"""절 CM126 escwhy 분석 (로컬, numpy 만): 판 디렉토리 <run>/esc/fNNN.npz (lib/anchorflow/fem_escwhy.py 기록) 에서
(1) 프레임별 이탈 입자 (프레임 변위가 같은 칸 (0.02 m) 입자 평균에서 벗어난 크기 dev, 위로 튄 변위) 를 찾고
(2) 걸음별 노드 이상 (걸음 끝 속도 |v1|, 걸음 속도 변화 |v1 − v0|) 상위 노드의 속성 (질량·확장 외부·바닥 sk/pm·고착 an·장벽 bG/bK·갱신 방식 (후퇴 오일러/HHT)·잔차) 을,
(3) 처음 이탈 입자의 받침 27 노드와 칸 가우스 점 (DP 상태·det X·특이값) 의 걸음별 값을 표로 낸다.
사용: python3 exe/escwhy_an.py <run 디렉토리> [--top 8] [--pn 0.48] [--frames a:b] [--part p1,p2] > report.txt
"""
import argparse
import glob
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--top", type=int, default=8)
ap.add_argument("--pn", type=float, default=0.48)
ap.add_argument("--frames", default="")
ap.add_argument("--part", default="")
ap.add_argument("--h", type=float, default=0.02)
a = ap.parse_args()

fs = sorted(glob.glob(os.path.join(a.run, "esc", "f*.npz")))
if a.frames:
    f0, f1 = [int(q) for q in a.frames.split(":")]
    fs = [f for f in fs if f0 <= int(os.path.basename(f)[1:4]) <= f1]
np.set_printoptions(precision=4, suppress=True, linewidth=200)


def cellmean_dev(x, d, h):
    c = np.floor(x / h).astype(np.int64)
    key = (c[:, 0] * 100000 + c[:, 1]) * 100000 + c[:, 2]
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    m = np.stack([np.bincount(inv, d[:, k], minlength=len(u)) for k in range(3)], 1) / cnt[:, None]
    return np.linalg.norm(d - m[inv], axis=1), cnt[inv]


def stencil(x, Ai, IP):
    y = x @ Ai.T
    b = np.floor(y - 0.5).astype(np.int64)
    fx = y - b
    b = b - IP[0:3]
    nb1, nb2 = IP[7], IP[8]
    ids, ws = [], []
    w = [np.stack([0.5 * (1.5 - fx[:, k]) ** 2, 0.75 - (fx[:, k] - 1) ** 2, 0.5 * (fx[:, k] - 0.5) ** 2], 1) for k in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ids.append(((b[:, 0] + i) * nb1 + (b[:, 1] + j)) * nb2 + (b[:, 2] + k))
                ws.append(w[0][:, i] * w[1][:, j] * w[2][:, k])
    return np.stack(ids, 1), np.stack(ws, 1), b


def nodepos(nid, IP, A):
    nb1, nb2 = IP[7], IP[8]
    g = np.stack([nid // (nb1 * nb2), (nid // nb2) % nb1, nid % nb2], 1) + IP[0:3]
    return g @ A.T


first = None
for fpath in fs:
    z = np.load(fpath)
    fr = int(os.path.basename(fpath)[1:4])
    x0, x1 = z["p0_x"], z["p1_x"]
    ok = np.isfinite(x1).all(1) & np.isfinite(x0).all(1)
    d = np.where(ok[:, None], x1 - x0, 0.0)
    dev, cnt = cellmean_dev(x0, d, a.h)
    up = d[:, 2]
    o = np.argsort(-dev)[:a.top]
    print(f"== f{fr:03d} rec {int(z['rec'])} K {int(z['K'])} 노드 상자 {int(z['IP'][9])} 가우스 점 {int(z['IP'][12])} 넘침 {int(z['IP'][13])} | 입자 dev 최대 {dev.max():.4f} 99.9% {np.quantile(dev, 0.999):.4f}"
          f" · 위 변위 최대 {up.max():.4f} (>0.02: {(up > 0.02).sum()}) · 비유한 {int((~ok).sum())}")
    for p in o[:4]:
        print(f"   입자 {p} dev {dev[p]:.4f} x0 {x0[p]} d {d[p]} 칸 입자 {cnt[p]} v0 {z['p0_v'][p]} v1 {z['p1_v'][p]} detF0 {np.linalg.det(z['p0_F'][p].reshape(3, 3)):.3f} detF1 {np.linalg.det(z['p1_F'][p].reshape(3, 3)):.3f}")
    if first is None and (dev.max() > 0.05 or up.max() > 0.05):
        first = (fr, o[:a.top])
    if "nid" not in z.files:
        continue
    nid, IP, A, Ai = z["nid"], z["IP"], z["A"], z["Ai"]
    K = int(z["K"]); dt = float(z["h"]) / K
    mn, W8 = z["mn"], z["W8"]
    pos = nodepos(nid.astype(np.int64), IP, A)
    mmed = np.median(mn)
    for k in range(K):
        V0, V1, U0, U1, Y = z["V0"][k], z["V1"][k], z["U0"][k], z["U1"][k], z["Y"][k]
        be = np.all(np.abs(U1 - Y) <= 1e-7 + 1e-6 * np.abs(Y), 1)
        dv = np.linalg.norm(V1 - V0, axis=1)
        sp = np.linalg.norm(V1, axis=1)
        g0 = np.linalg.norm(z["G0"][k], axis=1)
        ext = W8[k] >= 0
        print(f"   걸음 {k}: |v1| 최대 {sp.max():.3f} (질량 {mn[sp.argmax()]:.2e} = 중앙값 x{mn[sp.argmax()] / mmed:.2e}), |Δv| 최대 {dv.max():.3f}; 후퇴 오일러 노드 {be.sum()} / {len(nid)},"
              f" 외부 노드 {ext.sum()}, pm2 (고정) {(z['PM'][k] == 2).sum()}, 장벽 노드 (bK>0) {(z['BK'][k] > 0).sum()}, 잔차 노드 |g0| 최대 {g0.max():.3e}")
        for n in np.argsort(-dv)[:3]:
            print(f"      노드 {nid[n]} x {pos[n] + U0[n]} 바닥 위 {pos[n][2] + U0[n][2] - a.pn:+.4f} m {mn[n]:.2e} (x{mn[n] / mmed:.1e}) mqn {z['mqn'][n]:.2e} 외부 {int(W8[k][n] >= 0)} sk {z['SK'][k][n]:.3f} pm {z['PM'][k][n]}"
                  f" an {z['AN'][k][n]:.2e} kb {z['KB'][k][n]:.3f} ul {z['UL'][k][n]} bG {z['BG'][k][n]:.2e} bK {z['BK'][k][n]:.2e} meff {z['ME'][k][n]:.2e} BE {int(be[n])}"
                  f" v0 {V0[n]} v1 {V1[n]} a0 {z['A0'][k][n]} a1 {z['A1'][k][n]} y−Ũ {Y[n] - z['Ut'][k][n]} |g0| {g0[n]:.2e} f {z['Fn'][k][n]}")
    if first is not None and first[0] == fr:
        print(f"-- 처음 이탈 프레임 f{fr}: 받침 노드 걸음별 (입자 {first[1][:3]})")
        ids, ws, b = stencil(x0[first[1][:3]], Ai, IP)
        sidx = np.argsort(nid)
        for pi, p in enumerate(first[1][:3]):
            print(f"  입자 {p} x0 {x0[p]} 바닥 위 {x0[p][2] - a.pn:+.4f} d {d[p]} v0 {z['p0_v'][p]} v1 {z['p1_v'][p]} a_p0 {z['p0_pa'][p]}")
            for o_ in np.argsort(-ws[pi])[:27]:
                gid = ids[pi, o_]
                j = np.searchsorted(nid, gid, sorter=sidx)
                j = sidx[j] if j < len(nid) else -1
                if j < 0 or nid[j] != gid:
                    print(f"    노드 {gid} w {ws[pi, o_]:.3f} -- 질량 0 (기록 없음)")
                    continue
                row = [f"w {ws[pi, o_]:.3f} m x{mn[j] / mmed:.1e} 바닥 위 {pos[j][2] - a.pn:+.4f}"]
                for k in range(K):
                    be = bool(np.all(np.abs(z['U1'][k][j] - z['Y'][k][j]) <= 1e-7 + 1e-6 * np.abs(z['Y'][k][j])))
                    row.append(f"k{k}: 외 {int(W8[k][j] >= 0)} pm {z['PM'][k][j]} sk {z['SK'][k][j]:.2f} ul {z['UL'][k][j]} BE {int(be)} bK {z['BK'][k][j]:.1e} v1 {z['V1'][k][j]} |g0| {np.linalg.norm(z['G0'][k][j]):.1e}")
                print(f"    노드 {gid} " + " | ".join(row))
            # 이 입자 칸의 가우스 점
            cb = z["cb"]
            cc = b[pi]
            ci = np.where((cb == cc).all(1))[0]
            if len(ci):
                c = ci[0]
                for oc in range(8):
                    q = c * 8 + oc
                    st = [f"k{k}: st {int(z['QS'][k][q, 3])} detX {z['QS'][k][q, 4]:.3f} σ {z['QS'][k][q, :3]}" for k in range(K)]
                    print(f"    점 {q} V {z['Vq'][q]:.2e} m {z['mq'][q]:.2e} bx {z['bx'][q] if 'bx' in z.files else '-'} det F0 {np.linalg.det(z['FQ0'][q].reshape(3, 3)):.3f} | " + " | ".join(st))
