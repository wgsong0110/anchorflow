"""롤아웃 덤프를 **정답 대비 나란히** 보여주는 영상으로 만든다.

왼쪽 PG MPM(기준), 오른쪽은 --label 로 받는다. (겹쳐 보기 칸은 요청으로 제거했다.)
색은 두 번째 칸에서 **정답과의 거리**라, 어디서 틀리는지가 바로 보인다.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import os as _os
for _p in ("/home/wgsong/.fonts/NotoSansCJKkr-Regular.otf",
           _os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")):
    try: fm.fontManager.addfont(_p)
    except Exception: pass
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
import imageio.v2 as imageio

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sub", type=int, default=6000)
ap.add_argument("--fps", type=int, default=8)
ap.add_argument("--label", default="출력만 최적화", help="오른쪽 칸 이름")
ap.add_argument("--label_left", default="PG MPM (기준)", help="왼쪽 칸 이름")
ap.add_argument("--handle_mark", choices=["frame", "init"], default="frame",
                help="빨간 입자를 고르는 법. frame=그 프레임에 반경 안인 것, "
                     "init=**첫 프레임에 반경 안이던 것**(정체를 고정해 어디로 "
                     "갔는지 따라갈 수 있다)")
ap.add_argument("--handle_s", type=float, default=1.6, help="빨간 점 크기")
ap.add_argument("--mark", nargs="+", default=[],
                help="'프레임:이름' 꼴. 그 프레임부터 제목에 표시하고 그 "
                     "프레임에서는 테두리를 굵게 한다 (예: 27:바닥닿음 50:상승)")
ap.add_argument("--esc_r", type=float, default=0.0,
                help="--color esc 의 공 반지름. 0 이면 초기 위치에서 중심 "
                     "입자까지의 최대 거리")
ap.add_argument("--esc_s", type=float, default=6.0, help="빨간 점 크기")
ap.add_argument("--cell_s", type=float, nargs=2, default=(18.0, 10.0),
                help="--color cell0 에서 섞인 셀·손잡이 안 점의 크기. 섞인 셀은 "
                     "수가 적어(실측 25 개) 키우지 않으면 화면에서 사라진다")
ap.add_argument("--plane", choices=["xz", "xy", "yz"], default="xz",
                help="어느 평면으로 볼지. xy 가 위에서 내려다본 것이다")
ap.add_argument("--ctrl_R", type=float, default=0.0,
                help="손잡이 반경을 직접 준다 (0 이면 덤프의 값)")
ap.add_argument("--quiver", type=int, default=0,
                help="유효 격자점(입자가 든 사면체의 꼭짓점)의 이동 방향을 "
                     "오른쪽 칸에 화살표로 그린다. 그릴 개수")
ap.add_argument("--quiver_scale", type=float, default=0.0,
                help="0 이면 화살표 중앙 길이가 화면 폭의 4%% 가 되게 자동")
ap.add_argument("--color", choices=["none", "err", "z0", "r0", "pos0", "cell0",
                                    "detF", "trC", "normE", "esc", "znow"],
                default="none",
                help="입자 색. none=검정, err=정답과의 거리(오른쪽 칸만), "
                     "z0=초기 높이, r0=초기 중심에서의 거리, pos0=**초기 위치 "
                     "(x,y,z)를 RGB 로** (물질점이 어디서 와서 어디로 갔는지 "
                     "한눈에 보인다), cell0=**첫 프레임 "
                     "셀 구분** (손잡이 밖 / 섞인 셀 / 손잡이 안). z0/r0/cell0 "
                     "은 두 칸에 같은 색을 입혀 어디로 갔는지 맞대 볼 수 있다. "
                     "znow=**그 프레임의 z** (두 칸이 같은 색 범위), "
                     "esc=**마지막 프레임에 공을 벗어난 입자**를 빨갛게 "
                     "(중심 입자에서 반지름 밖 + 중심보다 위). 정체를 고정해 "
                     "처음부터 끝까지 같은 입자를 칠한다. detF=행렬식"
                     "(1 이 부피 보존), trC=우 코시-그린의 대각합"
                     "(3 이 변형 없음), normE=그린-라그랑주 변형률의 "
                     "프로베니우스 노름(0 이 변형 없음). 셋 다 프레임마다 다시 "
                     "칠한다")
a = ap.parse_args()

D = torch.load(a.dump, map_location="cpu", weights_only=False)
P, G = D["pred"].float().numpy(), D["gt"].float().numpy()
EXT = float(D["EXT"])
T, N, _ = P.shape
rng = np.random.default_rng(0)
sel = rng.permutation(N)[:min(a.sub, N)]
P, G = P[:, sel], G[:, sel]
err = np.linalg.norm(P - G, axis=-1) / EXT * 100
lo = np.minimum(P.reshape(-1, 3).min(0), G.reshape(-1, 3).min(0))
hi = np.maximum(P.reshape(-1, 3).max(0), G.reshape(-1, 3).max(0))
pad = 0.06 * float(np.linalg.norm(hi - lo))
print(f"[덤프] {D['tag']} t0={D['t0']}  {T} 프레임, 입자 {N} "
      f"(그리는 건 {len(sel)})  오차 중앙 {np.median(err):.3f}% 최대 {err.max():.3f}%",
      flush=True)

# 손잡이 위치와 반경 (없으면 그리지 않는다)
_CP = D.get("ctrl_pos")
if _CP is not None:
    _CP = np.asarray(_CP, dtype=np.float32)
_rr = a.ctrl_R if a.ctrl_R > 0 else D.get("ctrl_R")
if _rr is None:
    _rr = 0.15
    if _CP is not None:
        print("[경고] 덤프에 ctrl_R 이 없다 -- 0.15 로 그린다 (실제와 다를 수 "
              "있다). --ctrl_R 로 넘겨라", flush=True)
_R_CTRL = float(np.asarray(_rr).reshape(-1)[0])

M_ESC = None
if a.color == "esc":
    _X0e = np.asarray(D["x0"], dtype=np.float32)
    _c0e = _X0e.mean(0)
    _cid = int(np.linalg.norm(_X0e - _c0e, axis=-1).argmin())
    _re = (a.esc_r if a.esc_r > 0
           else float(np.linalg.norm(_X0e - _X0e[_cid], axis=-1).max()))
    _PF = D["pred"].float().numpy()[-1]
    _cen = _PF[_cid]
    _esc_full = (np.linalg.norm(_PF - _cen, axis=-1) > _re) & (_PF[:, 2] > _cen[2])
    M_ESC = _esc_full[sel]
    print(f"[벗어남] 중심 입자 {_cid}, 반지름 {_re:.4f}, 마지막 중심 z "
          f"{_cen[2]:.4f} -> 벗어난 입자 {int(_esc_full.sum())}/"
          f"{_esc_full.size} (그리는 표본에서 {int(M_ESC.sum())})", flush=True)

CRGB = None
if a.color == "pos0":
    _X0 = np.asarray(D["x0"], dtype=np.float32)[sel]
    _lo0, _hi0 = _X0.min(0), _X0.max(0)
    CRGB = (_X0 - _lo0) / np.maximum(_hi0 - _lo0, 1e-12)
    print(f"[pos0] 초기 위치를 RGB 로: x->빨강 {_lo0[0]:.3f}~{_hi0[0]:.3f}, "
          f"y->초록 {_lo0[1]:.3f}~{_hi0[1]:.3f}, "
          f"z->파랑 {_lo0[2]:.3f}~{_hi0[2]:.3f}", flush=True)

# 입자별 고정 색 (z0/r0). 두 칸이 같은 값을 쓰므로 대응이 보인다.
CVAL, CLAB, CMAP = None, "", "viridis"
if a.color == "cell0":
    _nd = D.get("nodes")
    if not _nd or len(_nd[0]) < 4:
        raise SystemExit("덤프에 입자별 셀 구분이 없다 -- 다시 덤프할 것")
    CVAL = np.asarray(_nd[0][3])[sel].astype(np.float32)
    CMAP = matplotlib.colors.ListedColormap(["0.75", "tab:orange", "red"])
    CLAB = ("셀 구분 (회색 손잡이 밖 / 주황 **섞인 셀** / 빨강 손잡이 안): "
            + " / ".join(f"{int((CVAL == k).sum())}" for k in (0, 1, 2)))
    print(f"[셀] {CLAB}", flush=True)
if a.color in ("z0", "r0"):
    X0 = np.asarray(D["x0"], dtype=np.float32)[sel]
    if a.color == "z0":
        CVAL, CLAB, CMAP = X0[:, 2], "초기 높이 z", "viridis"
    else:
        cen = X0.mean(0)
        CVAL = np.linalg.norm(X0 - cen, axis=-1)
        CLAB, CMAP = "초기 중심에서의 거리", "plasma"
    print(f"[색] {CLAB}  {CVAL.min():.4f} ~ {CVAL.max():.4f}", flush=True)

# 유효 격자점과 그 변위 (덤프에 있을 때만)
NODES = D.get("nodes")
QS = a.quiver_scale
QCLIP = None
if a.quiver and NODES:
    # 배율을 **중앙값**으로 잡으면 이상치 몇 개가 화면을 덮는다 (실측). 90 분위
    # 를 화면 폭의 6% 에 맞추고, 그보다 긴 화살표는 2 배에서 자른다.
    _ln = np.concatenate([
        np.linalg.norm(np.asarray(dp)[np.asarray(sp)], axis=-1)
        for _np_, dp, sp, *_ in NODES])
    _p90, _mx = float(np.percentile(_ln, 90)), float(_ln.max())
    if QS <= 0:
        QS = 0.06 * float(hi[0] - lo[0] + 2 * pad) / max(_p90, 1e-12)
    QCLIP = 2.0 * _p90
    print(f"[화살표] 유효 격자점 변위 90 분위 {_p90:.3e} 최대 {_mx:.3e}, "
          f"배율 {QS:.1f} 배, {a.quiver} 개 (길이는 {QCLIP:.3e} 에서 자른다)",
          flush=True)
elif a.quiver:
    print("[화살표] 덤프에 격자점이 없다 -- 그리지 않는다", flush=True)

# 첫 프레임에 손잡이 반경 안이던 입자 (정체 고정)
M_INIT = None
if _CP is not None:
    X0i = np.asarray(D["x0"], dtype=np.float32)[sel]
    M_INIT = np.zeros(len(sel), bool)
    for _k in range(_CP.shape[1]):
        M_INIT |= np.linalg.norm(X0i - _CP[int(D["t0"]), _k], axis=-1) < _R_CTRL
    print(f"[손잡이] 반경 {_R_CTRL:.4f}, 첫 프레임 반경 안 "
          f"{int(M_INIT.sum())} 개 (그리는 표본 기준)", flush=True)

DETF, _w = None, 1.0
_FK = {"detF": (0, 1.0, "det(F)"), "trC": (1, 3.0, "tr(C)  C = FᵀF"),
       "normE": (2, 0.0, "‖E‖_F  E = (C−I)/2")}
if a.color in _FK:
    _k, _mid, _nm = _FK[a.color]
    _fs = D.get("fscal")
    if _fs is None:
        raise SystemExit("덤프에 변형 스칼라가 없다 -- --ov_roll 로 다시 덤프")
    DETF = np.asarray(_fs)[:, sel, _k]
    if a.color == "normE":
        _w = float(np.percentile(DETF, 99)) or 1e-6
        CMAP, CLAB = "inferno", f"{_nm}  (0 ~ {_w:.4f})"
    else:
        _w = float(np.percentile(np.abs(DETF - _mid), 99)) or 1e-6
        CMAP, CLAB = "coolwarm", f"{_nm}  ({_mid:g} ± {_w:.4f})"
    CVAL = DETF[0]
    print(f"[{a.color}] 범위 {DETF.min():.5f} ~ {DETF.max():.5f}, "
          f"색 범위 {CLAB}", flush=True)

if a.color == "znow":
    _allz = np.concatenate([P[:, :, 2].ravel(), G[:, :, 2].ravel()])
    CMAP = "viridis"
    CVAL = G[0][:, 2]            # 프레임마다 아래에서 갈아끼운다
    CLAB = f"그 프레임의 z ({np.percentile(_allz,1):.2f} ~ "
    CLAB += f"{np.percentile(_allz,99):.2f})"
    print(f"[znow] 색 범위 {np.percentile(_allz,1):.4f} ~ "
          f"{np.percentile(_allz,99):.4f}", flush=True)
    _ZLO = float(np.percentile(_allz, 1)); _ZHI = float(np.percentile(_allz, 99))

VLO, VHI = (_ZLO, _ZHI) if a.color == "znow" else (0.0, 2.0) if a.color == "cell0" else (
    ((0.0, _w) if a.color == "normE" else
     (_FK[a.color][1] - _w, _FK[a.color][1] + _w)) if a.color in _FK else (
    (float(CVAL.min()), float(CVAL.max())) if CVAL is not None else (0.0, 1.0)))

_CID = D.get("ctrl_id")
_XF = _XG = None
if _CID is not None:
    _CID = np.asarray(_CID).reshape(-1).astype(int)
    _XF = D["pred"].float().numpy()      # 부분표본 전 전체 (색인이 전체 기준)
    _XG = D["gt"].float().numpy()
    print(f"[손잡이] 입자 {_CID.tolist()} -- 칸마다 자기 위치로 원을 그린다",
          flush=True)

frames = []
for t in tqdm(range(T), desc="렌더", ncols=80):
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 5.0), dpi=110)
    i, j = {"xz": (0, 2), "xy": (0, 1), "yz": (1, 2)}[a.plane]
    # 손잡이 반경 안에 든 입자를 **빨갛게** 칠한다. 기준 칸은 PG 위치로,
    # 오른쪽 칸은 그 칸의 위치로 각각 판정한다 (같은 중심·같은 반경).
    mG = mP = None
    if _CP is not None and a.handle_mark == "init":
        mG = mP = M_INIT
    elif _CP is not None:
        _ti0 = min(int(D["t0"]) + t, _CP.shape[0] - 1)
        mG = np.zeros(len(sel), bool); mP = np.zeros(len(sel), bool)
        for _k in range(_CP.shape[1]):
            _c = _CP[_ti0, _k]
            mG |= np.linalg.norm(G[t] - _c, axis=-1) < _R_CTRL
            mP |= np.linalg.norm(P[t] - _c, axis=-1) < _R_CTRL

    if a.color == "znow":
        CVAL = G[t][:, 2]
    _oG = slice(None) if mG is None else ~mG
    _cG = (CRGB[_oG] if CRGB is not None else
           ("0.25" if CVAL is None else CVAL[_oG]))
    ax[0].scatter(G[t][_oG, i], G[t][_oG, j], s=1.1, c=_cG, cmap=CMAP,
                  vmin=None if CVAL is None else VLO,
                  vmax=None if CVAL is None else VHI, linewidths=0)
    if mG is not None and mG.any():
        ax[0].scatter(G[t][mG, i], G[t][mG, j], s=a.handle_s, c="red",
                      linewidths=0)
    ax[0].set_title(f"{a.label_left}   손잡이 안 "
                    f"{0 if mG is None else int(mG.sum())}", fontsize=9)
    _oP = slice(None) if mP is None else ~mP
    # 손잡이 밖은 기준 칸과 같은 검은색이다. 구속이 들어간 자리만 빨강으로
    # 떠야 하니 오차 색칠을 걷어냈다 (요청).
    if a.color == "err":
        ax[1].scatter(P[t][_oP, i], P[t][_oP, j], s=1.1, c=err[t][_oP],
                      cmap="inferno", vmin=0,
                      vmax=float(np.percentile(err, 99)) or 1.0, linewidths=0)
    else:
        ax[1].scatter(P[t][_oP, i], P[t][_oP, j], s=1.1,
                      c=(CRGB[_oP] if CRGB is not None else
                         ("0.25" if CVAL is None else CVAL[_oP])), cmap=CMAP,
                      vmin=None if CVAL is None else VLO,
                      vmax=None if CVAL is None else VHI, linewidths=0)
    if mP is not None and mP.any():
        ax[1].scatter(P[t][mP, i], P[t][mP, j], s=a.handle_s, c="red",
                      linewidths=0)
    ax[1].set_title(f"{a.label} (평균 오차 {err[t].mean():.3f}% EXT)"
                    f"   손잡이 안 {0 if mP is None else int(mP.sum())}",
                    fontsize=9)
    # 손잡이를 그린다. 이게 없으면 구동이 들어갔는지 눈으로 확인할 수 없어
    # "손잡이가 없는 것 같다" 는 오해를 부른다 (덤프에는 늘 들어 있다).
    if _CID is not None:
        # 칸마다 **자기 입자**의 현재 위치를 중심으로 쓴다
        for _q, _X in ((ax[0], G[t]), (ax[1], P[t])):
            for _k in range(len(_CID)):
                _c = _XF[t][_CID[_k]] if _q is ax[1] else _XG[t][_CID[_k]]
                _q.add_patch(plt.Circle((_c[i], _c[j]), _R_CTRL, fill=False,
                                        ec="deepskyblue", lw=1.6, alpha=.9))
                _q.plot([_c[i]], [_c[j]], marker="x", ms=7, mew=2.0,
                        color="deepskyblue")
    elif _CP is not None:
        _ti = min(int(D["t0"]) + t, _CP.shape[0] - 1)
        for _k in range(_CP.shape[1]):
            _c = _CP[_ti, _k]
            for _q in ax:
                _q.add_patch(plt.Circle((_c[i], _c[j]), _R_CTRL,
                                        fill=False, ec="deepskyblue", lw=1.6,
                                        alpha=.9))
                _q.plot([_c[i]], [_c[j]], marker="x", ms=7, mew=2.0,
                        color="deepskyblue")

    if M_ESC is not None and M_ESC.any():
        for q, X in ((ax[0], G[t]), (ax[1], P[t])):
            q.scatter(X[M_ESC, i], X[M_ESC, j], s=a.esc_s, c="red",
                      linewidths=0, zorder=4)
    if a.color == "cell0":
        # 수가 적은 범주(섞인 셀 25 개, 손잡이 안 24 개)는 기본 크기로는 보이지
        # 않는다. 같은 색으로 **위에 덧그린다**.
        for q, X in ((ax[0], G[t]), (ax[1], P[t])):
            for _k, _c, _sz in ((2, "red", a.cell_s[1]),
                                (1, "tab:orange", a.cell_s[0])):
                _m = CVAL == _k
                if _m.any():
                    q.scatter(X[_m, i], X[_m, j], s=_sz, c=_c, linewidths=0,
                              zorder=3 + (_k == 1))
    for q in ax:
        q.set_xlim(lo[i] - pad, hi[i] + pad); q.set_ylim(lo[j] - pad, hi[j] + pad)
        q.set_aspect("equal"); q.set_xticks([]); q.set_yticks([])
    if a.quiver and NODES:
        _npz, _dpz, _spz, *_r = NODES[min(t, len(NODES) - 1)]
        _npz = np.asarray(_npz); _dpz = np.asarray(_dpz)
        _idx = np.nonzero(np.asarray(_spz))[0]
        if len(_idx) > a.quiver:
            _idx = _idx[np.linspace(0, len(_idx) - 1, a.quiver).astype(int)]
        _vq = _dpz[_idx][:, [i, j]]
        _lq = np.linalg.norm(_dpz[_idx], axis=-1, keepdims=True)
        _vq = _vq * np.minimum(1.0, QCLIP / np.maximum(_lq, 1e-30))
        ax[1].quiver(_npz[_idx, i], _npz[_idx, j],
                     _vq[:, 0] * QS, _vq[:, 1] * QS,
                     angles="xy", scale_units="xy", scale=1.0,
                     width=.0025, color="deepskyblue", alpha=.85)
    if CLAB:
        ax[0].set_title(ax[0].get_title() + f"   색 = {CLAB}", fontsize=10)
    _mk = ""
    for _m in a.mark:
        _f, _, _nm = _m.partition(":")
        if t >= int(_f):
            _mk += f"   [{_nm} {int(_f)}~]"
        if t == int(_f):
            for _q in ax:
                for _sp in _q.spines.values():
                    _sp.set_linewidth(3.0); _sp.set_color("tab:red")
    fig.suptitle(f"{D['tag']}{_mk}  (학습에 쓰지 않은 시드)   "
                 f"자기회귀 {t + 1}/{T} 프레임", fontsize=12)
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3]
    if buf.shape[0] % 2 or buf.shape[1] % 2:
        buf = buf[:buf.shape[0] // 2 * 2, :buf.shape[1] // 2 * 2]
    frames.append(buf.copy())
    plt.close(fig)

imageio.mimsave(a.out, frames, fps=a.fps, quality=8, macro_block_size=1)
print(f"[저장] {a.out}  {len(frames)} 프레임", flush=True)
