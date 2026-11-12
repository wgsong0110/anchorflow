"""wolf H9 증분 포텐셜 항별 기록 (결과 불변, 읽기만): 프레임 (= 걸음; 쪼갠 프레임은 마지막 걸음) 끝 해에서 항별 에너지를 재고 입자·칸에 나눠 <out>/ip/f<프레임>.npz.

항 (모두 그 프레임 마지막 걸음의 풀이 상태; HHT 풀이 변수 y (α 단계 위치), ŷ = 풀이가 쓴 관성 예측 yh, τ² = (1−α_f)β dt²/(1−α_m)):
  - 관성 (노드):   ½ m_n |y_n − ŷ_n|² / τ²                  (m_n = 노드 질량 s.mn)
  - 중력 (노드):   −m_n g·(U_n,끝 − U_n,걸음시작)             (노드 실제 변위; ŷ 에도 τ² g 가 들어 있음 -- 사용자 정의대로 따로 적음)
  - 탄성 (가우스 점): V_q ψ(X_q),  X_q = 걸음 끝 탄성 F (커밋 RM 뒤, fpi_q 의 Fe[18:27]), ψ = Hencky μ Σ (ln σ)² + ½ λ (Σ ln σ)²
  - 바닥 장벽 (표본점): 최종 실행 소스의 rt_bar_q 글자를 복제 (W̄·기준 높이 이어 받기 등 모든 고침 포함), 점별 에너지만 따로 버퍼에 (힘·헤시안·전역 표지 쓰기 줄 지움)
  - 면 정규화 (면): ghost6 gq_force 와 같은 식의 면별 값 -- 강성 ½ k w_F D(y)ᵀ M D(y) (k = s._gpk, wolf κ 0 이면 0) · 고스트 질량 ½ (k_M/τ²) w_F D(y−ŷ)ᵀ M D(y−ŷ),
                    w_F = 실행 면 가중 g5w (lightg3/lightg4: 1[P1X] + (1 − 1[P1X]) r κ λ), 면 표지 g3fm. 확장 B-스플라인은 제약이라 에너지 없음.
입자로 나누기:
  - 노드 항: 입자 B-스플라인 가중 w_pn 비례 (노드마다 Σ_p w_pn 으로 나눔)
  - 가우스 점 항: 그 팔분 칸 입자에 부피 비례 (입자 없는 팔분 칸 몫은 같은 칸 입자에 부피 비례)
  - 장벽: 표본점 (팔분 칸) 입자에 부피 비례
  - 면 항: 양쪽 칸에 반씩, 칸 안 입자에 부피 비례
설치: exe/wolfip_wrap.py (가장 바깥). 그래프 안에는 걸음 시작 U 사본 복사 하나만 더함 (rt_ga_step0 직전). 나머지는 step() 뒤 그래프 밖, 따로 컴파일한
모듈 (실행 모듈 최종 소스 + 이 모듈 커널; 장치 전역은 실행 모듈 값을 복사) 과 torch 로만.
"""
import json
import os
import re
import traceback

import numpy as np
import torch

ST = dict(out=".", nframe=0, ncall=0, err=None, bar_args={}, mod2=None, mod2_id=None, globals=0, info={})
WIP_SRC = r"""
// ---- wolfip: 면별 정규화 에너지 (gq_force 와 같은 식, 칸 c 의 +a 면) ----
__global__ void wip_gq_e(const int* IP, const int* cb, const unsigned char* fm, const float* fw, float k, float cm, const float* y, const float* yh, const float* y0, double* Ek, double* Em) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= IP[11]) return;
    for (int a=0;a<3;a++) { Ek[c*3+a] = 0.0; Em[c*3+a] = 0.0; }
    int m = fm[c];
    if (!m) return;
    int s0, s1; int base = gq_base(IP, cb, c, &s0, &s1);
    for (int q=0;q<3;q++) {
        float u[27], w[27];
        for (int o=0;o<27;o++) { int id = base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3); float yy = y[id*3+q]; u[o] = yy + (y0 ? y0[id*3+q] : 0.f); w[o] = yy - yh[id*3+q]; }
        for (int a=0;a<3;a++) {
            if (!(m & (1 << a))) continue;
            float wf = fw ? fw[c*3+a] : 1.f;
            float Dy[9], Dd[9], MD[9];
            for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
                int ie = gq_en(base, a, i, j, s0, s1);
                float ye = y[ie*3+q], ue = ye + (y0 ? y0[ie*3+q] : 0.f), we = ye - yh[ie*3+q];
                Dy[i*3+j] = -u[gq_cn(a,0,i,j)] + 3.f*u[gq_cn(a,1,i,j)] - 3.f*u[gq_cn(a,2,i,j)] + ue;
                Dd[i*3+j] = -w[gq_cn(a,0,i,j)] + 3.f*w[gq_cn(a,1,i,j)] - 3.f*w[gq_cn(a,2,i,j)] + we;
            }
            float s = 0.f; gq_MDM(Dy, MD); for (int t=0;t<9;t++) s += Dy[t]*MD[t];
            Ek[c*3+a] += 0.5*(double)k*(double)wf*(double)s;
            s = 0.f; gq_MDM(Dd, MD); for (int t=0;t<9;t++) s += Dd[t]*MD[t];
            Em[c*3+a] += 0.5*(double)cm*(double)wf*(double)s;
        }
    }
}
"""


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _body(src, name):
    m = re.search(r"__global__\s+void\s+(?:__launch_bounds__\(\d+\)\s+)?" + name + r"\s*\(", src)
    if not m:
        return None
    b = src.find("{", m.end())
    d, i = 0, b
    while i < len(src):
        if src[i] == "{":
            d += 1
        elif src[i] == "}":
            d -= 1
            if d == 0:
                break
        i += 1
    return src[m.start():i + 1], src[m.end():b].rsplit(")", 1)[0]


def bar_clone(src):
    """최종 소스의 rt_bar_q -> wip_bar_q: 점별 에너지 E[qq] 만 씀 (기울기·헤시안 원자 합, 전역 표지·점 스칼라 쓰기, 건너뛰기 줄 지움)."""
    r = _body(src, "rt_bar_q")
    assert r is not None, "rt_bar_q 없음"
    body, params = r
    drop = ("atomicAdd(&Gn[", "AF_BPG[qq]", "AF_BPH[qq] = h", "AF_BCF[qq] |= 1", "AF_SKIP")
    lines = [ln for ln in body.split("\n") if not any(d in ln for d in drop)]
    b = "\n".join(lines)
    assert b.count("blocksum_add(e, E);") == 1, "rt_bar_q blocksum_add(e, E) 줄"
    b = b.replace("blocksum_add(e, E);", "if (on) E[qq] = e;", 1)
    b = re.sub(r"\brt_bar_q\s*\(", "wip_bar_q(", b, count=1)
    names = [re.findall(r"(\w+)\s*$", p.strip())[0] for p in params.split(",")]
    return b, names


def _gsize(tp):
    if "*" in tp:
        return 8
    if "double" in tp or "long long" in tp or "size_t" in tp or "int64" in tp:
        return 8
    if "char" in tp:
        return 1
    if "half" in tp or "short" in tp:
        return 2
    return 4


GRE = re.compile(r"^[ \t]*(?:__device__|__constant__)[ \t]+(?!__forceinline__|inline|__noinline__)([A-Za-z_][\w \t\*]*?)[ \t\*]*\b([A-Za-z_]\w*)[ \t]*(?:\[[ \t]*(\d+)[ \t]*\])?[ \t]*(?:=[^;(]*)?;", re.M)


def _globals(src):
    out = []
    for m in GRE.finditer(src):
        tp, nm, n = m.group(1), m.group(2), m.group(3)
        full = m.group(0)
        tpx = tp + ("*" if re.search(r"\*\s*" + nm, full) else "")
        out.append((nm, _gsize(tpx) * (int(n) if n else 1)))
    return out


def _mod2(s):
    """실행 모듈 최종 소스 + wip 커널 을 따로 컴파일 (실행 모듈 id 마다 한 번)."""
    from . import fem_an_rt as ART
    mod = s.mod
    if ST["mod2_id"] == id(mod) and ST["mod2"] is not None:
        return ST["mod2"]
    src = mod.code
    assert src and "rt_bar_q(" in src and "gq_force(" in src and "gq_base(" in src, "실행 모듈 소스에 rt_bar_q·gq_force 없음"
    bc, names = bar_clone(src)
    ST["bar_params"] = names
    i = src.rstrip().rfind("}")
    code = src[:i] + "\n// ---- wolfip ----\n" + bc + "\n" + WIP_SRC + "\n}\n"
    opts = tuple(getattr(mod, "options", ("--std=c++14",)) or ("--std=c++14",))
    m2 = type(mod)(code=code, options=opts)                                         # 진짜 cupy RawModule (감싸개들의 소스 고침을 다시 타지 않게)
    m2.get_function("wip_bar_q"); m2.get_function("wip_gq_e")
    ST["mod2"], ST["mod2_id"] = m2, id(mod)
    ST["glist"] = [g for g in _globals(src)]
    ST["info"]["mod2_compiled"] = ST["info"].get("mod2_compiled", 0) + 1
    ST["info"]["globals"] = [g[0] for g in ST["glist"]]
    with open(os.path.join(ST["out"], "wip_bar_q.cu"), "w") as f_:
        f_.write(bc)
    return m2


def _copy_globals(s, m2):
    from . import fem_an_rt as ART
    n = 0
    bad = []
    for nm, sz in ST["glist"]:
        try:
            a = s.mod.get_global(nm)
            b = m2.get_global(nm)
        except Exception:
            bad.append(nm)
            continue
        b.copy_from_device(a, sz)
        n += 1
    ST["info"]["globals_copied"] = n
    ST["info"]["globals_missing"] = bad


def _launch(m2, name, n, args, b=128):
    from . import fem_an_rt as ART
    fn = m2.get_function(name)
    fn(((int(n) + b - 1) // b,), (b,), args)


def _bw(f):
    return torch.stack([0.5 * (1.5 - f) ** 2, 0.75 - (f - 1.0) ** 2, 0.5 * (f - 0.5) ** 2], dim=-1)


def measure(s, t):
    from . import fem_ghost6 as G6
    torch.cuda.synchronize()
    dev = s.dev
    fr = int(round(t / s.h)) + 1
    IPl = [int(q) for q in s.IP.cpu().tolist()]
    nn, nc, nq = IPl[9], IPl[11], IPl[12]
    nb = IPl[6:9]
    K = int(getattr(s, "last_K", s.K))
    sg = s.stages(K)[-1]
    tau = float(sg["tau"]); it2 = 1.0 / (tau * tau); thinv = float(sg["thinv"])
    d = torch.float64
    # ---- 노드 항 ----
    mn = s.mn[:nn].double()
    y = s.y[:nn].double(); yh = s.yh[:nn].double(); U1 = s.U[:nn].double(); U0 = s.wipU0[:nn].double()
    g = torch.tensor(s.g, device=dev, dtype=d)
    e_in_n = 0.5 * mn * ((y - yh) ** 2).sum(1) * it2
    e_gr_n = -mn * ((U1 - U0) @ g)
    # ---- 탄성 (가우스 점) ----
    X = s.Fe[:nq, 18:27].double().reshape(-1, 3, 3)
    Vq = s.Vq[:nq].double()
    sv = torch.linalg.svdvals(X).clamp_min(1e-30)
    ls = torch.log(sv)
    psi = float(s.mu) * (ls ** 2).sum(1) + 0.5 * float(s.lam) * ls.sum(1) ** 2
    detX = torch.linalg.det(X)
    okq = torch.isfinite(psi) & (Vq > 0)
    psi = torch.where(okq, psi, torch.zeros_like(psi))
    e_el_q = Vq * psi
    # ---- 장벽 (표본점) · 면 정규화 ----
    m2 = _mod2(s)
    _copy_globals(s, m2)
    Eq = torch.zeros(max(nq, 1), device=dev, dtype=d)
    ba = ST["bar_args"].get(K)
    names = ST["bar_params"]
    if ba is not None:
        a = list(ba)
        idx = {nm: i for i, nm in enumerate(names)}
        scr = torch.zeros(int(s.capN), device=dev, dtype=torch.float32)
        a[idx["U"]] = P(s.wipU0); a[idx["y"]] = P(s.y); a[idx["mode"]] = np.int32(1)
        a[idx["Gn"]] = P(scr); a[idx["Kn"]] = P(scr); a[idx["E"]] = P(Eq)
        _launch(m2, "wip_bar_q", int(s.capQc) * 8, tuple(a))
        ST["info"]["bar_thinv"] = float(a[idx["thinv"]]); ST["info"]["bar_ks"] = float(a[idx["ks"]]); ST["info"]["bar_dh"] = float(a[idx["dh"]])
    else:
        ST["info"].setdefault("bar_missing_K", []).append(K)
    k = float(getattr(s, "_gpk", 0.0) or 0.0)
    kM = float(getattr(s, "_g3kM", 0.0) or 0.0) if G6.MASS else 0.0
    cm = kM * it2
    fwp = G6.FWP(s)
    Y0 = P(s.g6u) if (G6.CONS and getattr(s, "g6u", None) is not None) else P(None)
    Ek = torch.zeros(max(nc, 1) * 3, device=dev, dtype=d); Em = torch.zeros_like(Ek)
    if getattr(s, "g3fm", None) is not None:
        _launch(m2, "wip_gq_e", int(s.capQc), (P(s.IP), P(s.cb), P(s.g3fm), fwp, np.float32(k), np.float32(cm), P(s.y), P(s.yh), Y0, P(Ek), P(Em)))
    torch.cuda.synchronize()
    e_bar_q = Eq[:nq]
    npen = int((~torch.isfinite(e_bar_q) | (e_bar_q >= 1e29)).sum())
    e_bar_q = torch.where(torch.isfinite(e_bar_q) & (e_bar_q < 1e29), e_bar_q, torch.zeros_like(e_bar_q))
    Ek = Ek[:nc * 3].view(nc, 3); Em = Em[:nc * 3].view(nc, 3)
    Ef = Ek + Em
    # ---- 입자 소속 ----
    N = s.N
    fx = s.fx.double(); pb = s.Np.long()
    cp_ = s.cidx[s.pcd.long()].long()
    oc = (fx[:, 0] >= 1).long() * 4 + (fx[:, 1] >= 1).long() * 2 + (fx[:, 2] >= 1).long()
    qp = cp_ * 8 + oc
    vol = s.vol.double()
    Vqp = torch.zeros(nc * 8, device=dev, dtype=d).index_add_(0, qp, vol)
    Vcp = torch.zeros(nc, device=dev, dtype=d).index_add_(0, cp_, vol)
    # 노드: B-스플라인 가중
    w = _bw(fx)
    W = (w[:, 0, :, None, None] * w[:, 1, None, :, None] * w[:, 2, None, None, :]).reshape(N, 27)
    o = torch.arange(27, device=dev)
    oi, oj, ok_ = o // 9, (o // 3) % 3, o % 3
    nid = ((pb[:, 0:1] + oi) * nb[1] + (pb[:, 1:2] + oj)) * nb[2] + (pb[:, 2:3] + ok_)
    Wn = torch.zeros(nn, device=dev, dtype=d).index_add_(0, nid.reshape(-1), W.reshape(-1))
    Wd = torch.where(Wn > 0, Wn, torch.ones_like(Wn))

    def node_share(en):
        sh = (W * (en / Wd)[nid]).sum(1)
        return sh, float(en[Wn <= 0].sum())

    def q_share(eq):
        eqf = torch.zeros(nc * 8, device=dev, dtype=d); eqf[:eq.numel()] = eq
        has = Vqp > 0
        sh = torch.where(has[qp], eqf[qp] * vol / torch.where(has, Vqp, torch.ones_like(Vqp))[qp], torch.zeros_like(vol))
        emp = torch.where(has, torch.zeros_like(eqf), eqf).view(nc, 8).sum(1)
        hc = Vcp > 0
        sh = sh + emp[cp_] * vol / torch.where(hc, Vcp, torch.ones_like(Vcp))[cp_]
        return sh, float(emp[~hc].sum()), float(emp.sum())

    def cell_share(ec):
        hc = Vcp > 0
        return ec[cp_] * vol / torch.where(hc, Vcp, torch.ones_like(Vcp))[cp_], float(ec[~hc].sum())

    p_in, un_in = node_share(e_in_n)
    p_gr, un_gr = node_share(e_gr_n)
    p_el, un_el, emp_el = q_share(e_el_q)
    p_bar, un_bar, emp_bar = q_share(e_bar_q)
    # 면 -> 양쪽 칸 반씩
    n3 = [nb[0] - 2, nb[1] - 2, nb[2] - 2]
    cid = s.cidx[:n3[0] * n3[1] * n3[2]].long().view(*n3)
    cbx = s.cb[:nc].long()
    reg_c = torch.zeros(nc, device=dev, dtype=d)
    reg_c.index_add_(0, torch.arange(nc, device=dev), 0.5 * Ef.sum(1))
    nbad = 0
    for ax in range(3):
        sel = Ef[:, ax] != 0
        if not bool(sel.any()):
            continue
        cc = torch.nonzero(sel).squeeze(1)
        x2 = cbx[cc].clone(); x2[:, ax] += 1
        inb = (x2[:, 0] < n3[0]) & (x2[:, 1] < n3[1]) & (x2[:, 2] < n3[2])
        c2 = torch.full_like(cc, -1)
        c2[inb] = cid[x2[inb, 0], x2[inb, 1], x2[inb, 2]]
        okc = (c2 >= 0) & (c2 < nc)
        nbad += int((~okc).sum())
        reg_c.index_add_(0, c2[okc], 0.5 * Ef[cc[okc], ax])
        reg_c.index_add_(0, cc[~okc], 0.5 * Ef[cc[~okc], ax])                      # 이웃 칸 못 찾은 면 (없어야 함): 자기 칸에 (수 기록)
    p_reg, un_reg = cell_share(reg_c)
    el_c = torch.zeros(nc, device=dev, dtype=d).index_add_(0, torch.arange(nq, device=dev) // 8, e_el_q)
    psi_p = psi[qp.clamp_max(nq - 1)] if nq > 0 else torch.zeros_like(vol)
    # ---- 원래 입자 순서로 저장 ----
    def orig(a):
        o_ = torch.empty_like(a); o_[s.perm] = a
        return o_.float().cpu().numpy()

    os.makedirs(os.path.join(ST["out"], "ip"), exist_ok=True)
    np.savez(os.path.join(ST["out"], "ip", f"f{fr:03d}.npz"), e_in=orig(p_in), e_el=orig(p_el), e_gr=orig(p_gr), e_bar=orig(p_bar), e_reg=orig(p_reg), psi=orig(psi_p),
             el_c=el_c.float().cpu().numpy(), reg_c=reg_c.float().cpu().numpy(), bar_c=torch.zeros(nc, device=dev, dtype=d).index_add_(0, torch.arange(nq, device=dev) // 8, e_bar_q).float().cpu().numpy(),
             ncell_p=Vcp.gt(0).sum().cpu().numpy())
    fwv = (s.g5w[:nc * 3].view(nc, 3) if getattr(s, "g5w", None) is not None else torch.ones(nc, 3, device=dev))
    fm = s.g3fm[:nc].long() if getattr(s, "g3fm", None) is not None else torch.zeros(nc, dtype=torch.long, device=dev)
    face = torch.stack([(fm >> b_) & 1 for b_ in range(3)], 1).bool()
    rec = dict(f=fr, t=float(t), K=K, tau=tau, it2=it2, thinv=thinv, nn=nn, nc=nc, nq=nq, N=N,
               node=dict(E_in=float(e_in_n.sum()), E_gr=float(e_gr_n.sum()), unassigned_in=un_in, unassigned_gr=un_gr, mn_sum=float(mn.sum())),
               gauss=dict(E_el=float(e_el_q.sum()), E_bar=float(e_bar_q.sum()), n_pen=npen, n_inv=int((detX <= 0).sum()), n_nonfin=int((~okq & (Vq > 0)).sum()),
                          empty_oct_el=emp_el, empty_oct_bar=emp_bar, unassigned_el=un_el, unassigned_bar=un_bar, Vq_sum=float(Vq.sum())),
               face=dict(E_k=float(Ek.sum()), E_m=float(Em.sum()), k=k, kM=kM, cm=cm, nface=int(face.sum()), nface_wpos=int((face & (fwv > 0)).sum()),
                         fw_mean=float(fwv[face].mean()) if bool(face.any()) else None, fw_max=float(fwv[face].max()) if bool(face.any()) else None,
                         unassigned=un_reg, nbad_neighbor=nbad, fwp=int(fwp) != 0, cons=int(G6.CONS)),
               part=dict(E_in=float(p_in.sum()), E_el=float(p_el.sum()), E_gr=float(p_gr.sum()), E_bar=float(p_bar.sum()), E_reg=float(p_reg.sum())),
               mu=float(s.mu), lam=float(s.lam), cdp=float(getattr(s, "cdp", 0.0)), g=list(s.g))
    with open(os.path.join(ST["out"], "wolfip.jsonl"), "a") as f_:
        f_.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def install(out):
    from . import fem_an_rt as ART
    ST["out"] = out
    R = ART.RTSolver
    oalloc, ok, orec, ostep = R.alloc, R.k, R.record, R.step

    def alloc(s, *a, **kw):
        r = oalloc(s, *a, **kw)
        s.wipU0 = torch.zeros(int(s.capN), 3, device=s.U.device, dtype=torch.float32)
        ST["bar_args"].clear()
        return r

    def record(s, t_frame, K=None, *a, **kw):
        s._wip_recK = int(K or s.K)
        return orec(s, t_frame, K, *a, **kw)

    def k(s, name, n, args, b=128):
        if name == "rt_ga_step0" and getattr(s, "wipU0", None) is not None and getattr(s, "_pkc", None) is None:
            s.dcopy(s.wipU0, s.U)                                                       # 걸음 시작 U (마지막 걸음 것이 남음)
        if name == "rt_bar_q":                                                          # 이 K 그래프의 장벽 인자 (모드는 측정 때 1 로 덮어씀)
            ST["bar_args"][getattr(s, "_wip_recK", s.K)] = tuple(args)
        return ok(s, name, n, args, b)

    def step(s, t):
        redo = ostep(s, t)
        ST["ncall"] += 1
        try:
            if bool(torch.isfinite(s.x[:1]).all()) and not (s.ga and getattr(s, "ga_first", False)):
                measure(s, t)
                ST["nframe"] += 1
        except Exception as ex:
            ST["err"] = repr(ex)
            with open(os.path.join(out, "wolfip_err.txt"), "a") as f_:
                f_.write(f"t {t}: {ex!r}\n{traceback.format_exc()}\n")
        if ST["ncall"] <= 3 or ST["ncall"] % 10 == 0:
            _dump()
        return redo

    R.alloc, R.k, R.record, R.step = alloc, k, record, step
    print(f"[wolfip] 증분 포텐셜 항별 기록 켬 -> {out}/ip, wolfip.jsonl", flush=True)


def _dump():
    try:
        d = {k_: v for k_, v in ST.items() if k_ not in ("mod2", "bar_args", "glist")}
        d["bar_K"] = list(ST["bar_args"].keys())
        with open(os.path.join(ST["out"], "wolfipinfo.json"), "w") as f_:
            json.dump(d, f_, ensure_ascii=False, default=str)
    except Exception:
        pass
