"""절 CM48 묶음 ghost2: 절 CM47 ghost penalty (lib/anchorflow/fem_ghost1.py) 검사 + γ 바꿈. 감싸개 인자 없으면 이 모듈을 안 부름 = 소스·그래프 그대로.

정의는 절 CM47 그대로 (같은 CUDA 소스 GP_SRC·같은 훅 자리): E_g = ½ k Σ_F Σ_{t,t'} (M⊗M)_{tt'} D_t·D_t', k = γ_a h, γ_a = (2μ + λ) · g (g = 환경 AF_G2_GAMMA, 절 CM47 은 1e-4).
바뀐 것 (결과 불변 쪽만):
  - g 를 인자로 (AF_G2_GAMMA). g = 0 이면 커널은 그대로 다 돌고 더하는 값만 0 (훅·버퍼·그래프 모양이 결과를 안 바꾸는지 보는 대조) -- 이때만 결정적 판 (--det) 허용.
  - 훅이 부른 커널 인자를 기억 (s._g2args) -> check() 가 그래프 밖에서 같은 인자로 '훅 켬 − 훅 끔' 을 재서 벌점 몫이 제 버퍼에 들어가는지 확인.
검사 (check, 결과 불변 -- 장치 상태는 읽기만, 출력은 새 버퍼):
  A. 면 집합: 커널 목록 (gp_cut·gp_face) 과 torch 로 따로 만든 목록 (cidx·cb 에서 26 이웃·축 이웃) 이 같은지, 경계 칸 수.
  B. 에너지·기울기·헤시안 곱: 커널 (gp_force E·f, gp_mv) 과 torch f64 참값 (따로 짠 D·M⊗M; 기울기는 autograd 로도) 의 상대 차.
  C. 유한차분: (E(y+εv) − E(y−εv)) / (f(y)·2εv), (f(y+εv) − f(y−εv)) / (G 2εv).
  D. 블록 대각: gp_force B 대각 vs 단위 벡터 헤시안 곱 (표본 노드 48).
  E. 선탐색 후보 에너지 (gp_ls·gp_lsfin): 합성 ŷ·η·δ·sk·pm (pm 0·1·2) 의 y_c = ŷ + T(η + c δ) 참값 에너지와, i0 = 2 판, gate 판.
  F. 옛 실행기 Φ 몫 (gp_phi) = gp_force E.
  H. 훅 자리: 기억한 요소·곱·선탐색·Φ 커널 인자로 '훅 켬 − 훅 끔' (k 를 잠깐 1e3 배로) = 벌점 커널 값인지 (기울기·블록 대각·에너지·곱·후보 에너지·Φ).
"""
import json
import math
import os

import numpy as np
import torch

from .fem_ghost1 import GP_SRC, _append                                                   # 절 CM47 커널 그대로

GAMMA_A = float(os.environ.get("AF_G2_GAMMA", "1e-4"))
ST = dict(on=False, said=False, nk={})
ELEM = ("rt_elem", "aq_elem", "ao_elem", "ao_elem_s")
MV = ("rt_mv", "rt_mv2", "rt_mv2s", "ao_mv_q", "ao_mv_h", "ao_mv_c", "ao_mv_s")
REC = ELEM + MV + ("rt_elem_lsN", "en_phi_q")
MM9 = np.array([[6, 13, 1], [13, 54, 13], [1, 13, 6]], dtype=np.float64) / 120.0
CL = (-1.0, 3.0, -3.0, 1.0)


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _k_of(s):
    if getattr(s, "_gpk", None) is None:
        A = s.A.double().reshape(3, 3).cpu()
        h = abs(float(torch.linalg.det(A))) ** (1.0 / 3.0)
        kap = 2.0 * float(s.mu) + float(s.lam)
        s._gpk = GAMMA_A * kap * h
        s._gph, s._gpkap = h, kap
        print(f"[절 CM48 ghost] γ_a = (2μ+λ)·{GAMMA_A:g} (μ {float(s.mu):.6g}, λ {float(s.lam):.6g}, 2μ+λ {kap:.6g}), h {h:.6g} m, k = γ_a h = {s._gpk:.6g}", flush=True)
    return s._gpk


def _alloc(s):
    cap = int(s.capQc)
    if getattr(s, "gpcut", None) is None or s.gpcut.numel() < cap:
        s.gpcut = torch.zeros(cap, device=s.dev, dtype=torch.int32)
        s.gpfl = torch.zeros(3 * cap, device=s.dev, dtype=torch.int32)
    if getattr(s, "gpcnt", None) is None:
        s.gpcnt = torch.zeros(4, device=s.dev, dtype=torch.int32)
        s.gpGQ = torch.zeros(4, device=s.dev, dtype=torch.float64)


def _cnt(name):
    ST["nk"][name] = ST["nk"].get(name, 0) + 1


def install():
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    from . import fem_an_rt_v0 as V0
    ST["on"] = True
    _obs, _oas = ART.build_src, AORT._src
    ART.build_src = lambda *a, **kw: _append(_obs(*a, **kw))
    AORT._src = lambda *a, **kw: _append(_oas(*a, **kw))
    f32 = np.float32

    def wrap_k(cls):
        _ok = cls.k

        def k(s, name, n, args, b=128):
            if name in REC and not getattr(s, "_g2chk", False):
                if getattr(s, "_g2args", None) is None:
                    s._g2args = {}
                s._g2args[name] = tuple(args)
            r = _ok(s, name, n, args, b)
            if getattr(s, "_gpin", False) or name.startswith("gp_"):
                return r
            s._gpin = True
            try:
                _hook(s, _ok, name, args)
            finally:
                s._gpin = False
            return r
        cls.k = k

    def _go(s, _ok, nm, n, args, b):
        _cnt(nm)
        _ok(s, nm, n, args, b)

    def _hook(s, _ok, name, args):
        if name == "rt_gauss":
            _k_of(s)
            if getattr(s, "det", False):
                assert s._gpk == 0.0, "ghost: 결정적 판 (det) 은 g = 0 대조만 (원자 합 순서)"
            assert getattr(s, "pkmod", None) is None, "ghost: 지속형 판은 안 고침"
            _alloc(s)
            cq = int(s.capQc)
            _go(s, _ok, "gp_zero", 1, (P(s.gpcnt),), 1)
            _go(s, _ok, "gp_cut", cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.gpcut), P(s.gpcnt)), 128)
            _go(s, _ok, "gp_face", 3 * cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.gpcut), P(s.gpcnt), P(s.gpfl)), 128)
            if not ST["said"]:
                ST["said"] = True
                print("[절 CM48 ghost] 켬: 경계 칸·면 목록 (rt_gauss 뒤), 요소·곱·에너지 커널 뒤 G 몫", flush=True)
            return
        if getattr(s, "gpcnt", None) is None:
            return
        k = f32(s._gpk)
        nf = 3 * int(s.capQc)
        G = (P(s.IP), P(s.cb), P(s.gpcnt), P(s.gpfl), k)
        if name in ELEM:
            if name == "rt_elem":
                mode = int(args[8]); y, E, f_, B = args[5], args[9], args[10], args[13]
                _go(s, _ok, "gp_force", nf, G + (y, f_ if mode & 1 else P(None), B if mode & 8 else P(None), E if mode & 4 else P(None)), 128)
            elif name == "aq_elem":
                _go(s, _ok, "gp_force", nf, G + (args[5], args[10], args[13], P(None)), 128)
            else:
                mode = int(args[9])
                _go(s, _ok, "gp_force", nf, G + (args[5], args[10] if mode & 1 else P(None), args[13] if mode & 8 else P(None), P(None)), 128)
        elif name in MV:
            _go(s, _ok, "gp_mv", nf, G + (args[6], args[7]), 128)
        elif name == "rt_bd":
            _go(s, _ok, "gp_force", nf, G[:4] + (k, P(s.y), P(None), args[6], P(None)), 128)
        elif name == "rt_elem_lsN":
            assert len(args) >= 19, "ghost: rt_elem_lsN (i0·gate 없는 옛 판) 은 안 고침"
            yh, eta, dl, sk, pm, n0, n1, n2, SC, NC, i0, gate = args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[15], args[16], args[17], args[18]
            _go(s, _ok, "gp_ls", nf, G[:4] + (yh, eta, dl, sk, pm, n0, n1, n2, P(s.gpGQ), gate), 128)
            _go(s, _ok, "gp_lsfin", 1, (SC, NC, i0, gate, P(s.gpGQ), k), 1)
        elif name == "en_phi_q":
            _go(s, _ok, "gp_phi", nf, G + (args[6], args[19], args[20]), 128)
        elif name == "rt_elem_ls":
            raise RuntimeError(f"ghost: {name} 경로는 안 고침")

    for cls in (ART.RTSolver, V0.RTSolver):
        wrap_k(cls)


# ------------------------------------------------------------------------------------------------ 검사 (그래프 밖, 결과 불변)
def _skip0(s):
    import cupy as cp
    for mod in (s._mods() if hasattr(s, "_mods") else [s.mod]):
        try:
            g = mod.get_global("AF_SKIP")
            cp.ndarray((1,), dtype=cp.int32, memptr=g)[0] = 0
        except Exception:
            pass


def ref_struct(s):
    """torch 로 따로 만든 경계 칸·면 목록과 면 36 노드 번호 (gp_ids 와 같은 배치: l 층 x 가로 i, j)."""
    IP = [int(q) for q in s.IP.cpu().tolist()]
    nb = IP[6:9]
    n = [nb[0] - 2, nb[1] - 2, nb[2] - 2]
    nc = IP[11]
    dev = s.dev
    cidx = s.cidx[:n[0] * n[1] * n[2]].long().view(*n)
    act = cidx >= 0
    cb = s.cb[:nc * 3].long().view(nc, 3)
    pad = torch.nn.functional.pad((~act).double()[None, None], (1, 1, 1, 1, 1, 1), value=1.0)
    mx = torch.nn.functional.max_pool3d(pad, 3, stride=1)[0, 0]
    cutg = act & (mx > 0)
    cut = cutg[cb[:, 0], cb[:, 1], cb[:, 2]]
    fcs, fas = [], []
    for a in range(3):
        x = cb.clone()
        x[:, a] += 1
        inb = x[:, a] < n[a]
        c2 = torch.full((nc,), -1, dtype=torch.long, device=dev)
        xi = x[inb]
        c2[inb] = cidx[xi[:, 0], xi[:, 1], xi[:, 2]]
        ok = c2 >= 0
        ok2 = torch.zeros_like(ok)
        ok2[ok] = cut[ok] | cut[c2[ok]]
        idx = torch.nonzero(ok2).squeeze(1)
        fcs.append(idx)
        fas.append(torch.full_like(idx, a))
    fc, fa = torch.cat(fcs), torch.cat(fas)
    nf = fc.numel()
    ids = torch.empty(nf, 4, 9, dtype=torch.long, device=dev)
    L = torch.arange(4, device=dev)
    I = torch.arange(3, device=dev)
    for a in range(3):
        m = fa == a
        if not bool(m.any()):
            continue
        t1 = 1 if a == 0 else 0
        t2 = 1 if a == 2 else 2
        X = cb[fc[m]][:, None, None, None, :].expand(-1, 4, 3, 3, 3).clone()
        X[..., a] += L[None, :, None, None]
        X[..., t1] += I[None, None, :, None]
        X[..., t2] += I[None, None, None, :]
        ids[m] = ((X[..., 0] * nb[1] + X[..., 1]) * nb[2] + X[..., 2]).reshape(-1, 4, 9)
    return dict(IP=IP, nc=nc, ncut=int(cut.sum()), fkey=torch.sort(fc * 3 + fa).values, ids=ids, nf=nf, cutkey=cb[cut], fc=fc, fa=fa, cb=cb)


def _MM(dev):
    return torch.tensor(np.kron(MM9, MM9), device=dev, dtype=torch.float64)


def ref_D(R, y):
    y3 = y.reshape(-1, 3).double()
    D = 0.0
    for l in range(4):
        D = D + CL[l] * y3[R["ids"][:, l, :]]
    return D                                                                                  # (nf, 9, 3)


def ref_E(R, y, k, MM):
    D = ref_D(R, y)
    return 0.5 * k * torch.einsum("fta,ts,fsa->", D, MM, D)


def ref_G(R, v, k, MM, nnode):
    D = ref_D(R, v)
    MD = torch.einsum("ts,fsa->fta", MM, D)
    out = torch.zeros(nnode, 3, dtype=torch.float64, device=v.device)
    for l in range(4):
        out.index_add_(0, R["ids"][:, l, :].reshape(-1), (k * CL[l] * MD).reshape(-1, 3))
    return out


def _rel(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    d = float((a - b).norm())
    nb_ = float(b.norm())
    return d / nb_ if nb_ > 0 else (0.0 if d == 0 else float("inf"))


def _T(v, sk, pm, nrm):
    """Tapply pm 0·1·2 (torch f64)."""
    vn = (v * nrm).sum(-1, keepdim=True)
    o0 = sk[:, None] * v
    o1 = sk[:, None] * (v - vn * nrm) + vn * nrm
    o = torch.where((pm == 0)[:, None], o0, o1)
    return torch.where((pm == 2)[:, None], torch.zeros_like(o), o)


def check(s, fi, out):
    """프레임 fi 걸음 뒤 (그래프 밖) 검사 -> dict (out 에 g2fd.jsonl 한 줄)."""
    from . import fem_an_rt as ART
    torch.cuda.synchronize()
    _skip0(s)
    dev = s.dev
    rs = dict(f=int(fi))
    k0 = float(s._gpk)
    kt = k0 if k0 > 0 else 1.0                                                                  # g = 0 판도 커널 검사는 k = 1 로
    capN, capQc = int(s.capN), int(s.capQc)
    nf_thr = 3 * capQc
    MM = _MM(dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(1234 + int(fi))
    R = ref_struct(s)
    nnode = R["IP"][9]
    cnt = [int(q) for q in s.gpcnt.cpu().tolist()]
    kf = torch.sort(s.gpfl[:cnt[1]].long()).values
    rs.update(ncut_k=cnt[0], ncut_r=R["ncut"], nface_k=cnt[1], nface_r=R["nf"], face_same=bool(kf.numel() == R["fkey"].numel() and bool((kf == R["fkey"]).all())), nnode=nnode, nc=R["nc"], k=k0, kt=kt)
    G = (P(s.IP), P(s.cb), P(s.gpcnt), P(s.gpfl))
    f32 = np.float32

    def kforce(y, k=kt, want=(True, True, True)):
        f_ = torch.zeros(capN * 3, device=dev, dtype=torch.float64)
        B_ = torch.zeros(capN * 9, device=dev, dtype=torch.float32)
        E_ = torch.zeros(2, device=dev, dtype=torch.float64)
        s.k("gp_force", nf_thr, G + (f32(k), P(y), P(f_) if want[0] else P(None), P(B_) if want[1] else P(None), P(E_) if want[2] else P(None)), 128)
        torch.cuda.synchronize()
        return f_, B_, float(E_[0])

    def kmv(v, k=kt):
        o = torch.zeros(capN * 3, device=dev, dtype=torch.float32)
        s.k("gp_mv", nf_thr, G + (f32(k), P(v), P(o)), 128)
        torch.cuda.synchronize()
        return o

    y = s.y.reshape(-1).contiguous()
    ymax = float(y[:nnode * 3].abs().max()) if nnode > 0 else 0.0
    sc = max(ymax, 1e-4)
    v = torch.randn(capN * 3, device=dev, generator=gen, dtype=torch.float32) * sc
    v[nnode * 3:] = 0
    # B. 에너지·기울기·헤시안 곱
    fk, Bk, Ek = kforce(y)
    Er = float(ref_E(R, y[:nnode * 3], kt, MM))
    yd = y[:nnode * 3].double().reshape(-1, 3).clone().requires_grad_(True)
    ga = torch.autograd.grad(ref_E(R, yd.reshape(-1), kt, MM), yd)[0]
    gr = ref_G(R, y[:nnode * 3], kt, MM, nnode)
    mk = kmv(v)
    mr = ref_G(R, v[:nnode * 3], kt, MM, nnode)
    rs.update(E_k=Ek, E_r=Er, E_rel=abs(Ek - Er) / max(abs(Er), 1e-300), f_rel=_rel(fk[:nnode * 3], gr), f_auto_rel=_rel(gr, ga), mv_rel=_rel(mk[:nnode * 3], mr))
    # C. 유한차분
    yp = (y + 1e-2 * v).contiguous()
    ym = (y - 1e-2 * v).contiguous()
    fp, _, Ep = kforce(yp)
    fm, _, Em = kforce(ym)
    dy = (yp - ym).contiguous()
    rs["fdE_ratio"] = (Ep - Em) / float((fk * dy.double()).sum()) if float((fk * dy.double()).sum()) != 0 else None
    rs["fdH_rel"] = _rel(fp - fm, kmv(dy).double())
    # D. 블록 대각
    nodes = torch.unique(R["ids"].reshape(-1))
    if nodes.numel() > 0:
        pick = nodes[torch.randint(0, nodes.numel(), (48,), device=dev, generator=gen)]
        dd = []
        for nd in pick.tolist():
            e = torch.zeros(capN * 3, device=dev, dtype=torch.float32)
            a_ = 0
            e[nd * 3 + a_] = 1.0
            o = kmv(e)
            dd.append((float(o[nd * 3 + a_]), float(Bk[nd * 9 + a_ * 4])))
        dd = np.array(dd)
        rs["diag_rel"] = float(np.abs(dd[:, 0] - dd[:, 1]).max() / max(np.abs(dd[:, 0]).max(), 1e-300))
    # E. 선탐색 후보 (합성)
    nrm = torch.tensor([0.6, 0.0, 0.8], device=dev, dtype=torch.float64)
    yh = y.clone()
    eta = (torch.randn(capN * 3, device=dev, generator=gen) * sc).float()
    dl = (torch.randn(capN * 3, device=dev, generator=gen) * sc).float()
    sk = (torch.rand(capN, device=dev, generator=gen) * 0.7 + 0.3).float()
    pm = torch.randint(0, 3, (capN,), device=dev, generator=gen).int()
    GQ = torch.zeros(4, device=dev, dtype=torch.float64)
    ls_err = []
    for i0, gate_v in ((0, None), (2, None), (0, 1.0), (0, 0.0)):
        SCb = torch.zeros(64, device=dev, dtype=torch.float64)
        gt = None if gate_v is None else torch.tensor([gate_v], device=dev, dtype=torch.float64)
        s.k("gp_ls", nf_thr, G + (P(yh), P(eta), P(dl), P(sk), P(pm), f32(0.6), f32(0.0), f32(0.8), P(GQ), P(gt)), 128)
        s.k("gp_lsfin", 1, (P(SCb), np.int32(10), np.int32(i0), P(gt), P(GQ), f32(kt)), 1)
        torch.cuda.synchronize()
        got = SCb[40:50].cpu().numpy()
        for i in range(10):
            c = 0.5 ** i
            if (gate_v is not None and gate_v != 0.0) or i < i0:
                exp = 0.0
            else:
                e3 = eta[:nnode * 3].double().reshape(-1, 3) + c * dl[:nnode * 3].double().reshape(-1, 3)
                yc = yh[:nnode * 3].double().reshape(-1, 3) + _T(e3, sk[:nnode].double(), pm[:nnode], nrm)
                exp = float(ref_E(R, yc.reshape(-1), kt, MM))
            ls_err.append(abs(got[i] - exp) / max(abs(exp), 1e-300) if exp != 0 else abs(got[i]))
        rs[f"ls_GQ_left_{i0}_{gate_v}"] = float(GQ.abs().sum())
        GQ.zero_()
    rs["ls_rel_max"] = float(max(ls_err))
    # F. Φ 몫
    MO = torch.ones(1, device=dev, dtype=torch.int32)
    PH = torch.zeros(2, device=dev, dtype=torch.float64)
    s.k("gp_phi", nf_thr, G + (f32(kt), P(y), P(MO), P(PH)), 128)
    torch.cuda.synchronize()
    rs["phi_rel"] = abs(float(PH[1]) - Ek) / max(abs(Ek), 1e-300)
    # H. 훅 자리 ('훅 켬 − 훅 끔' = 벌점 커널)
    if not getattr(s, "det", False) and getattr(s, "_g2args", None):
        s._g2chk = True
        try:
            rs["hook"] = _hook_check(s, R, MM, nnode, kt, gen, sc)
        finally:
            s._g2chk = False
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if out:
        with open(os.path.join(out, "g2fd.jsonl"), "a") as fo:
            fo.write(json.dumps(rs) + "\n")
    return rs


def _tensor_by_ptr(s, ptr):
    for v_ in vars(s).values():
        if torch.is_tensor(v_) and v_.numel() > 0 and int(v_.data_ptr()) == int(ptr):
            return v_
    return None


def _hook_check(s, R, MM, nnode, kt, gen, sc):
    dev = s.dev
    capN, capQ = int(s.capN), int(s.capQc) * 8
    nf_thr = 3 * int(s.capQc)
    f32 = np.float32
    Gk = lambda k: (P(s.IP), P(s.cb), P(s.gpcnt), P(s.gpfl), f32(k))
    k_old = s._gpk
    kb = kt * 1e3
    o = {}
    try:
        s._gpk = kb
        for name, args in list(s._g2args.items()):
            try:
                if name in ELEM:
                    res = []
                    for raw in (True, False):
                        a = list(args)
                        f_ = torch.zeros(capN * 3, device=dev, dtype=torch.float64)
                        B_ = torch.zeros(capN * 9, device=dev, dtype=torch.float32)
                        E_ = torch.zeros(64, device=dev, dtype=torch.float64)
                        Ht = _tensor_by_ptr(s, a[11])
                        if Ht is None and int(a[11]) != 0:
                            raise RuntimeError("H 버퍼 못 찾음")
                        H_ = torch.empty_like(Ht) if Ht is not None else None
                        a[10], a[11], a[13] = P(f_), P(H_), P(B_)
                        if name == "rt_elem":
                            a[9] = P(E_)
                        if name in ("aq_elem", "ao_elem", "ao_elem_s") and len(a) > 14:
                            a[14] = P(None)
                        if name == "aq_elem" and len(a) > 18 and int(a[18]) != 0:
                            qt = _tensor_by_ptr(s, a[18])
                            a[18] = P(torch.zeros_like(qt)) if qt is not None else P(None)
                        s._gpin = raw
                        s.k(name, capQ, tuple(a))
                        s._gpin = False
                        torch.cuda.synchronize()
                        res.append((f_, B_, float(E_[0])))
                        del H_
                    mode = int(args[8]) if name == "rt_elem" else (11 if name == "aq_elem" else int(args[9]))
                    fg = torch.zeros(capN * 3, device=dev, dtype=torch.float64)
                    Bg = torch.zeros(capN * 9, device=dev, dtype=torch.float32)
                    Eg = torch.zeros(2, device=dev, dtype=torch.float64)
                    s.k("gp_force", nf_thr, Gk(kb) + (args[5], P(fg), P(Bg), P(Eg)), 128)
                    torch.cuda.synchronize()
                    d = dict(mode=mode)
                    if mode & 1:
                        d["f_rel"] = _rel(res[1][0] - res[0][0], fg)
                    if mode & 8:
                        d["B_rel"] = _rel((res[1][1] - res[0][1]).double(), Bg.double())
                    if name == "rt_elem" and mode & 4:
                        d["E_rel"] = abs((res[1][2] - res[0][2]) - float(Eg[0])) / max(abs(float(Eg[0])), 1e-300)
                    o[name] = d
                elif name in MV:
                    vv = (torch.randn(capN * 3, device=dev, generator=gen) * sc).float()
                    vv[nnode * 3:] = 0
                    outs = []
                    for raw in (True, False):
                        a = list(args)
                        oo = torch.zeros(capN * 3, device=dev, dtype=torch.float32)
                        a[6], a[7] = P(vv), P(oo)
                        s._gpin = raw
                        s.k(name, capQ, tuple(a))
                        s._gpin = False
                        torch.cuda.synchronize()
                        outs.append(oo)
                    og = torch.zeros(capN * 3, device=dev, dtype=torch.float32)
                    s.k("gp_mv", nf_thr, Gk(kb) + (P(vv), P(og)), 128)
                    torch.cuda.synchronize()
                    o[name] = dict(mv_rel=_rel((outs[1] - outs[0]).double(), og.double()))
                elif name == "rt_elem_lsN":
                    yh = s.y.reshape(-1).clone()
                    eta = (torch.randn(capN * 3, device=dev, generator=gen) * sc * 1e-2).float()
                    dl = (torch.randn(capN * 3, device=dev, generator=gen) * sc * 1e-2).float()
                    sk = (torch.rand(capN, device=dev, generator=gen) * 0.7 + 0.3).float()
                    pm = torch.randint(0, 3, (capN,), device=dev, generator=gen).int()
                    scs = []
                    for raw in (True, False):
                        a = list(args)
                        SCb = torch.zeros(64, device=dev, dtype=torch.float64)
                        a[5], a[6], a[7], a[8], a[9] = P(yh), P(eta), P(dl), P(sk), P(pm)
                        a[10], a[11], a[12] = f32(0.6), f32(0.0), f32(0.8)
                        a[15], a[16], a[17], a[18] = P(SCb), np.int32(10), np.int32(0), P(None)
                        s._gpin = raw
                        s.k(name, capQ, tuple(a))
                        s._gpin = False
                        torch.cuda.synchronize()
                        scs.append(SCb[40:50].cpu().numpy())
                    nrm = torch.tensor([0.6, 0.0, 0.8], device=dev, dtype=torch.float64)
                    errs = []
                    for i in range(10):
                        c = 0.5 ** i
                        e3 = eta[:nnode * 3].double().reshape(-1, 3) + c * dl[:nnode * 3].double().reshape(-1, 3)
                        yc = yh[:nnode * 3].double().reshape(-1, 3) + _T(e3, sk[:nnode].double(), pm[:nnode], nrm)
                        ex = float(ref_E(R, yc.reshape(-1), kb, MM))
                        errs.append(abs((scs[1][i] - scs[0][i]) - ex) / max(abs(ex), 1e-300))
                    o[name] = dict(ls_rel_max=float(max(errs)), ls_fin=bool(np.isfinite(scs[0]).all()))
                elif name == "en_phi_q":
                    phs = []
                    for raw in (True, False):
                        a = list(args)
                        MO = torch.ones(1, device=dev, dtype=torch.int32)
                        PH = torch.zeros(2, device=dev, dtype=torch.float64)
                        a[19], a[20] = P(MO), P(PH)
                        s._gpin = raw
                        s.k(name, capQ, tuple(a))
                        s._gpin = False
                        torch.cuda.synchronize()
                        phs.append(float(PH[1]))
                    Eg = torch.zeros(2, device=dev, dtype=torch.float64)
                    s.k("gp_force", nf_thr, Gk(kb) + (args[6], P(None), P(None), P(Eg)), 128)
                    torch.cuda.synchronize()
                    o[name] = dict(phi_rel=abs((phs[1] - phs[0]) - float(Eg[0])) / max(abs(float(Eg[0])), 1e-300))
            except Exception as ex:
                o[name] = dict(err=repr(ex)[:300])
            finally:
                s._gpin = False
    finally:
        s._gpk = k_old
    return o


def face_change(s, prev):
    """프레임 사이 경계 칸·면 집합 바뀜 (절대 칸 좌표 = 상자 bmin + cb). prev = 앞 프레임 (keys) -> (dict, 지금 keys)."""
    R = ref_struct(s)
    IP = R["IP"]
    off = torch.tensor(IP[0:3], device=s.dev, dtype=torch.long)
    enc = lambda X: ((X[:, 0] + 100000) * 200000 + (X[:, 1] + 100000)) * 200000 + (X[:, 2] + 100000)
    ck = torch.unique(enc(R["cutkey"] + off))
    fk = torch.unique(enc(R["cb"][R["fc"]] + off) * 3 + R["fa"])
    ak = torch.unique(enc(R["cb"] + off))
    d = dict(ncut=int(ck.numel()), nface=int(fk.numel()), nact=int(ak.numel()))
    if prev is not None:
        for nm, cur, old in (("cut", ck, prev[0]), ("face", fk, prev[1]), ("act", ak, prev[2])):
            inter = int(torch.isin(cur, old).sum())
            uni = cur.numel() + old.numel() - inter
            d["jac_" + nm] = inter / uni if uni else 1.0
            d["new_" + nm] = cur.numel() - inter
    return d, (ck, fk, ak)
