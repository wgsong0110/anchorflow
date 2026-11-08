"""절 CM69 묶음 lightg (11-08 위임): 가벼운 노드 튐 -- 강성 (+ 고스트 질량) 벌점의 면 가중을 면 집합 대신 노드 가벼움으로 연속 가중.

원인 (절 CM64·CM67): 튐은 저질량 문턱 m̄ (= mlow × 첫 프레임 노드 질량 중앙, mlow 1e-3) 바로 위 (질량 비 1e-3~5e-2, 87 %) 의 풀린 노드가
  거의 빈 칸에서 한 걸음에 0.5~3 m 움직이는 것. 그 자리는 P1X (3x3x3 열림 면) 가 강성 벌점을 빼는 겉 부스러기 면 (절 CM55.2).
  그 면에 면 집합 (방식 0·A·B) 으로 벌점을 걸면 ficus 줄기·잎 면과 wolf 겉층까지 걸려 두 씬이 깨짐 (절 CM53·CM55).

정의 (새 문턱·클램프 없음; 기존 스케일 m̄ 만):
  노드 가벼움 l_n = m̄ / (m_n + m̄)   (m_n = 그 프레임 P2G 노드 질량 (원래 질량), m_n > 0 노드만; m_n ≪ m̄ -> 1, m_n = m̄ -> ½, m_n ≫ m̄ -> m̄/m_n -> 0)
    = 절 CM67 (1) 덧질량 몫 w = m_n/(m_n + m̄) 의 여수 (1 − w): 관성 예측에 묶이는 몫 = 그 노드가 제 질량으로 못 받치는 몫.
  면 가벼움 λ_F = max_{n ∈ 면 36 노드, m_n > 0} l_n  (AF_LG_AGG 0, 기본; 1 = 질량 가중 평균 Σ m_n l_n / Σ m_n -- 진단 비교용)
    면 36 노드 = 그 면의 법선 3 차 차분 스텐실 (칸 27 노드 + 이웃 칸 바깥 층 9) = 그 면 위에서 받침이 겹치는 이차 B-스플라인 기저 전부.
  벌점 = G'' 의 P1X 벌점 (면 집합 = 3x3x3 열림 두꺼운 칸끼리, γ̄ 1e-1) 그대로 + 새 몫 γ̄_L Σ_{F ∈ 후보 면} λ_F (강성 + 고스트 질량 면 에너지)
    후보 면 = 절 CM47 면 (축 이웃 두 칸 활성, 하나 이상 경계 칸) 에서 구속 노드 면 빼기 (@g4x 1) 뒤 -- ghost penalty 가 정의되는 면 전부 (집합 고르기 없음).
    -> 실행에서는 ghost6 면 표지를 후보 면 전부 (AF_G4_THIN 0) 로 두고 면 가중 fw_F = 1[F ∈ P1X] + (γ̄_L / γ̄) λ_F 를 ghost6 의 가중 버퍼 g5w 에 씀
       (ghost6 커널: 강성 k fw D·MD + 질량 (k_M/τ²) fw ... -- 힘·곱·블록 대각·선탐색·Φ 모두 같은 fw).
  무거운 노드뿐인 면 (ficus 줄기, wolf 몸통) 은 λ_F ≈ m̄/m_min ≪ 1, 문턱 아래·바로 위 가벼운 노드가 낀 면은 λ_F ≥ ~0.5.
  m̄ = 0 인 씬 (레고·ficus: --mlow 없음) 은 λ ≡ 0 이라 정의상 바뀌는 것 없음 -> 아무 커널도 안 띄우고 P1X 경로 (면 표지 방식 1) 그대로.
  가중은 프레임마다 rt_vn (P2G 노드 질량 끝) 뒤에 한 번 (걸음 안에서는 상수).

진단 (결과 불변): AF_LG_DG=1 -> lg.jsonl (프레임 끝: 후보 면·P1X 면 수, λ 분포 (최댓값·질량 가중 평균), 면의 가장 가벼운 노드로 가른 면 무리별 λ),
  AF_LG_FD=프레임들 -> lgfd.jsonl (그래프 밖: 가중 커널 대 torch, 기울기·헤시안 곱·선탐색 에너지 유한차분, 훅 자리).
"""
import json
import os

import numpy as np
import torch

GL = float(os.environ.get("AF_LG_G", "0"))                                                 # γ̄_L
AGG = int(os.environ.get("AF_LG_AGG", "0"))                                                 # 0 최댓값 · 1 질량 가중 평균
DG = int(os.environ.get("AF_LG_DG", "0"))
FD = {int(q) for q in os.environ.get("AF_LG_FD", "").split(",") if q.strip()}
ST = dict(on=None, log=None, out=".", nw=0, said=False)

LG_SRC = r'''
// ---- 절 CM69 lightg: 면 가중 = 1[P1X 면] + r · 면 36 노드 가벼움 (lib/anchorflow/fem_lightg.py) ----
// 두꺼운 칸 (3x3x3 열림, ghost6 gq_thk thin 1 과 같은 식): 닫힌 27 이웃에 경계 아닌 활성 칸이 있음
__global__ void gq_lgthk(const int* IP, const int* cb, const int* cidx, const unsigned char* cut, unsigned char* thk) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int n0 = IP[6]-2, n1 = IP[7]-2, n2 = IP[8]-2;
    int x = cb[c*3], y = cb[c*3+1], z = cb[c*3+2], k = 0;
    for (int dx=-1;dx<=1 && !k;dx++) for (int dy=-1;dy<=1 && !k;dy++) for (int dz=-1;dz<=1 && !k;dz++) {
        int X = x+dx, Y = y+dy, Z = z+dz;
        if (X < 0 || Y < 0 || Z < 0 || X >= n0 || Y >= n1 || Z >= n2) continue;
        int c2 = cidx[(X*n1 + Y)*n2 + Z];
        if (c2 >= 0 && !cut[c2]) k = 1;
    }
    thk[c] = (unsigned char)k;
}
// 면 (칸 c, 축 a) 36 노드 (m > 0) 의 가벼움 l = mbar/(m + mbar): 최댓값 lx · 질량 가중 평균 lm · 가장 가벼운 노드 질량 mi; fw = 1[두 칸 두꺼움] + r (agg ? lm : lx)
__global__ void gq_lgw(const int* IP, const int* cb, const int* cidx, const unsigned char* fm, const unsigned char* thk, const float* mn, float mbar, float r, int agg,
                       float* fw, float* wlx, float* wlm, float* wmi) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int m = fm[c];
    int n[3] = {IP[6]-2, IP[7]-2, IP[8]-2};
    int s0, s1; int base = gq_base(IP, cb, c, &s0, &s1);
    float l0 = 0.f, sm0 = 0.f, sl0 = 0.f, mi0 = 3.0e38f;
    if (m) {
        for (int o=0;o<27;o++) {
            float q = mn[base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3)];
            if (q > 0.f) { float l = mbar/(q + mbar); l0 = fmaxf(l0, l); sm0 += q; sl0 += q*l; mi0 = fminf(mi0, q); }
        }
    }
    for (int a=0;a<3;a++) {
        float w = 0.f, lx = 0.f, lm = 0.f, mi = 0.f;
        if (m & (1 << a)) {
            int x2[3] = {cb[c*3], cb[c*3+1], cb[c*3+2]}; x2[a] += 1;
            int c2 = (x2[a] < n[a]) ? cidx[(x2[0]*n[1] + x2[1])*n[2] + x2[2]] : -1;
            float l1 = l0, sm = sm0, sl = sl0; mi = mi0;
            for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
                float q = mn[gq_en(base, a, i, j, s0, s1)];
                if (q > 0.f) { float l = mbar/(q + mbar); l1 = fmaxf(l1, l); sm += q; sl += q*l; mi = fminf(mi, q); }
            }
            lx = l1; lm = (sm > 0.f) ? sl/sm : 0.f;
            int ind = (c2 >= 0 && thk[c] && thk[c2]) ? 1 : 0;
            w = (float)ind + r*(agg ? lm : lx);
        }
        fw[c*3+a] = w; wlx[c*3+a] = lx; wlm[c*3+a] = lm; wmi[c*3+a] = mi;
    }
}
'''


def _append(src):
    if "gq_lgw(" in src:
        return src
    assert "gq_base(" in src and "gq_en(" in src, "lightg: ghost6 소스 (gq_base·gq_en) 가 먼저 붙어야 함"
    i = src.rstrip().rfind("}")
    return src[:i] + LG_SRC + "\n}\n"


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _on(s):
    return float(getattr(s, "mthr", 0.0) or 0.0) > 0.0 and GL > 0.0


def _alloc(s):
    cap = int(s.capQc)
    if getattr(s, "lgthk", None) is None or s.lgthk.numel() < cap:
        s.lgthk = torch.zeros(cap, device=s.dev, dtype=torch.uint8)
        s.lglx = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
        s.lglm = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
        s.lgmi = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)


def ratio():
    from . import fem_ghost6 as G6
    return GL / float(G6.GAMMA)


def weights(s, ok):
    """프레임마다 rt_vn 뒤: 두꺼운 칸 -> 면 가중 (ghost6 g5w 에)."""
    _alloc(s)
    cq = int(s.capQc)
    ok(s, "gq_lgthk", cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.g3cut), P(s.lgthk)), 128)
    ok(s, "gq_lgw", cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.g3fm), P(s.lgthk), P(s.mn), np.float32(s.mthr), np.float32(ratio()), np.int32(AGG),
                         P(s.g5w), P(s.lglx), P(s.lglm), P(s.lgmi)), 128)
    ST["nw"] += 1


def install(out):
    """실행기 runpy 순간 (다른 훅 뒤 = 가장 바깥) 에 부름."""
    from . import fem_an_rt as ART
    from . import fem_an_rt_v0 as V0
    from . import fem_ao_rt as AORT
    from . import fem_wmrx as WRX
    from . import fem_ghost6 as G6
    assert G6.ST["on"], "lightg: ghost6 (P1X) 가 켜져 있어야 함"
    assert G6.THIN == 1 and G6.WMODE == 0 and not G6.CONS, ("lightg: 바탕은 P1X (면 방식 1, 가중 없음, 증분 판)", G6.THIN, G6.WMODE, G6.CONS)
    ST.update(out=out)
    if DG:
        ST["log"] = open(os.path.join(out, "lg.jsonl"), "w")
    o_b, o_s = ART.build_src, AORT._src
    ART.build_src = lambda *a, **kw: _append(o_b(*a, **kw))
    AORT._src = lambda *a, **kw: _append(o_s(*a, **kw))

    def wrap_k(cls):
        _ok = cls.k

        def k(s, name, n, args, b=128):
            if name.startswith("gq_") or getattr(s, "_lgin", False):
                return _ok(s, name, n, args, b)
            if name == "rt_gauss" and ST["on"] is None:                                       # 첫 프레임: 씬 결정 (m̄ > 0 이면 후보 면 전부 + 가중, 아니면 P1X 그대로)
                ST["on"] = _on(s)
                if ST["on"]:
                    G6.THIN = 0
                    G6.WMODE = 99                                                              # g5w 포인터를 커널에 넘기게 (gq5_w 의 99 = 1; rt_vn 뒤 이 모듈이 덮어씀)
                print(f"[절 CM69 lightg] 씬 결정: m̄ {float(getattr(s, 'mthr', 0.0) or 0.0):.4g}, γ̄_L {GL:g} (γ̄ {G6.GAMMA:g}, r {ratio():g}), 모음 {'최댓값' if AGG == 0 else '질량 가중 평균'} -> "
                      f"{'후보 면 전부 + 가벼움 가중' if ST['on'] else '아무것도 안 함 (P1X 그대로)'}", flush=True)
            r = _ok(s, name, n, args, b)
            if ST["on"] and name == "rt_vn":
                s._lgin = True
                try:
                    weights(s, _ok)
                finally:
                    s._lgin = False
            return r
        cls.k = k

    for cls in (ART.RTSolver, V0.RTSolver):
        wrap_k(cls)

    dep = dict(d=0)

    def outer(fn):
        def w(s, t, h, *a, **kw):
            dep["d"] += 1
            try:
                res = fn(s, t, h, *a, **kw)
            finally:
                dep["d"] -= 1
            if dep["d"] == 0 and ST["on"]:
                fi = int(round(t / h)) + 1
                try:
                    if DG:
                        diag(s, fi)
                    if fi in FD:
                        fdcheck(s, fi, out)
                except Exception:
                    import traceback
                    with open(os.path.join(out, "lg_err.txt"), "a") as f_:
                        f_.write(f"f{fi}\n" + traceback.format_exc()[-3000:] + "\n")
            return res
        return w
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)
    if hasattr(ART, "split_step"):
        ART.split_step = outer(ART.split_step)
    print(f"[절 CM69] lightg 켬 -- 면 가중 1[P1X] + (γ̄_L/γ̄) max 노드 가벼움 m̄/(m+m̄), γ̄_L {GL:g}, 모음 {AGG}, 진단 {DG}, 검사 프레임 {sorted(FD)}", flush=True)


def _q(v, qs=(0.1, 0.5, 0.9)):
    if v.numel() == 0:
        return None
    return [round(float(x), 5) for x in torch.quantile(v.float()[:1000000], torch.tensor(qs, device=v.device)).tolist()]


@torch.no_grad()
def diag(s, fi):
    """프레임 끝 (그래프 밖): 면 수·가벼움 분포. 면 무리 = 그 면 36 노드 중 가장 가벼운 노드 질량 m_min 으로: m_min < m̄ (고정 노드 낌) · m̄ ≤ m_min < 50 m̄ (질량 비 1e-3~5e-2 = 튐 노드 띠) · 그 위.
    (무리 경계는 보고용 분류일 뿐, 정의에 안 들어감.)"""
    from . import fem_ghost6 as G6
    torch.cuda.synchronize()
    nc = int(s.IP[11])
    keys = G6.mask_keys(s.g3fm, nc)
    mt = float(s.mthr)
    nn = int(s.IP[9])
    mn = s.mn[:nn]
    med = float(mn[mn > 0].median()) if bool((mn > 0).any()) else 0.0
    rec = dict(f=fi, ncand=int(keys.numel()), mbar=mt, mbar_med=(mt / med if med > 0 else None), nw=ST["nw"])
    if keys.numel():
        fw, lx, lm, mi = s.g5w[keys], s.lglx[keys], s.lglm[keys], s.lgmi[keys]
        r = ratio()
        ind = torch.round(fw - r * (lm if AGG else lx))
        p1 = ind > 0.5
        rec.update(np1x=int(p1.sum()), sum_l=float((lm if AGG else lx).sum()), ind_err=float((fw - r * (lm if AGG else lx) - ind).abs().max()),
                   lx_np1x=_q(lx[~p1]), lm_np1x=_q(lm[~p1]), lx_p1x=_q(lx[p1]), lm_p1x=_q(lm[p1]))
        grp = {}
        for nm, sel in (("fix", mi < mt), ("band", (mi >= mt) & (mi < 50 * mt)), ("heavy", mi >= 50 * mt)):
            grp[nm] = dict(n=int(sel.sum()), np1x=int((sel & p1).sum()), lx=_q(lx[sel]), lm=_q(lm[sel]), lx_mean=(float(lx[sel].mean()) if bool(sel.any()) else None))
        rec["grp"] = grp
    ST["log"].write(json.dumps(rec) + "\n")
    ST["log"].flush()


def _ref(s, G6):
    """torch 참값: 후보 면 (구속 노드 면 뺀 뒤) 36 노드, P1X 지시, 가벼움 (최댓값·질량 가중 평균), 가중."""
    R0 = G6.ref_struct3(s, 0)
    if G6.FIXX and getattr(s, "fixd", None) is not None and R0["nf"] > 0:
        keep = ~s.fixd.bool()[R0["ids"].reshape(R0["ids"].shape[0], -1)].any(1)
        R0 = dict(R0)
        R0.update(fc=R0["fc"][keep], fa=R0["fa"][keep], ids=R0["ids"][keep], nf=int(keep.sum()), fkey=torch.sort(R0["fc"][keep] * 3 + R0["fa"][keep]).values)
    R1 = G6.ref_struct3(s, 1)
    key = R0["fc"] * 3 + R0["fa"]
    ind = torch.isin(key, R1["fkey"]).double()
    nn = int(s.IP[9])
    mn = s.mn[:nn].double()
    ids = R0["ids"].reshape(R0["ids"].shape[0], -1)
    mm = mn[ids]
    mb = float(s.mthr)
    l = torch.where(mm > 0, mb / (mm + mb), torch.zeros_like(mm))
    lx = l.max(1).values
    sm = mm.clamp_min(0).sum(1)
    lm = torch.where(sm > 0, (mm.clamp_min(0) * l).sum(1) / sm.clamp_min(1e-300), torch.zeros_like(sm))
    R0 = dict(R0)
    R0["w"] = ind + ratio() * (lm if AGG else lx)
    return R0, key, ind, lx, lm


def fdcheck(s, fi, out):
    """프레임 fi 걸음 뒤 (그래프 밖, 결과 불변): (A) 가중 커널 대 torch, (B) 에너지 -> 기울기 유한차분, 기울기 -> 헤시안 곱 유한차분 (커널끼리) + torch 참값,
    (C) 선탐색 후보 에너지 대 같은 점의 에너지 커널 + 선탐색 에너지의 c 방향 유한차분 대 기울기, (D) 훅 자리 (실행기 커널 '훅 켬 − 훅 끔' 대 가중 참값)."""
    from . import fem_ghost6 as G6
    from . import fem_ghost2 as G2
    torch.cuda.synchronize()
    G2._skip0(s)
    s._g3chk = True
    s._lgin = True
    dev = s.dev
    f32 = np.float32
    rs = dict(f=int(fi))
    try:
        IP = [int(q) for q in s.IP.cpu().tolist()]
        nnode, nc = IP[9], IP[11]
        capN, capQc = int(s.capN), int(s.capQc)
        k0, kM = float(s._gpk), float(s._g3kM)
        kt = k0 if k0 > 0 else 1.0
        it2 = float(getattr(s, "_g3it2", 0.0)) or 1.0
        cmt = (kM if kM > 0 else 1.0) * it2
        MM = G2._MM(dev)
        gen = torch.Generator(device=dev)
        gen.manual_seed(9100 + int(fi))
        fm, fw = s.g3fm, s.g5w
        # (A) 가중
        R0, key, ind, lx, lm = _ref(s, G6)
        kk = G6.mask_keys(fm, nc)
        rs.update(nface=int(kk.numel()), nface_ref=int(R0["nf"]), set_eq=bool(kk.numel() == R0["fkey"].numel() and bool((kk == R0["fkey"]).all())),
                  np1x=int(ind.sum()), r=ratio(), k=k0, kM=kM, kt=kt, cmt=cmt)
        if R0["nf"]:
            wk = fw[key].double()
            rs.update(w_abs=float((wk - R0["w"]).abs().max()), lx_abs=float((s.lglx[key].double() - lx).abs().max()), lm_abs=float((s.lglm[key].double() - lm).abs().max()),
                      w_q=_q(R0["w"]), lx_q=_q(lx))
        # (B) 유한차분
        y = s.y.reshape(-1).contiguous()
        yh = s.yh.reshape(-1).contiguous()
        sc = max(float(y[:nnode * 3].abs().max()) if nnode else 0.0, 1e-4)
        d = torch.zeros(capN * 3, device=dev)
        d[:nnode * 3] = torch.randn(nnode * 3, device=dev, generator=gen) * sc

        def fo(yv):
            f_ = torch.zeros(capN * 3, device=dev, dtype=torch.float64)
            E_ = torch.zeros(2, device=dev, dtype=torch.float64)
            s.k("gq_force", capQc, (P(s.IP), P(s.cb), P(fm), P(fw), f32(kt), f32(cmt), P(yv), P(yh), P(None), P(f_), P(None), P(E_)), 128)
            torch.cuda.synchronize()
            return f_, float(E_[0])

        def mv(v):
            o_ = torch.zeros(capN * 3, device=dev)
            s.k("gq_mv", capQc, (P(s.IP), P(s.cb), P(fm), P(fw), f32(kt + cmt), P(v), P(o_)), 128)
            torch.cuda.synchronize()
            return o_
        f0, E0 = fo(y)
        n3 = nnode * 3
        fr = G6.ref_Gw(R0, y[:n3].double(), kt, MM, nnode) + G6.ref_Gw(R0, (y - yh)[:n3].double(), cmt, MM, nnode)
        Er = float(G6.ref_Ew(R0, y[:n3].double(), kt, MM)) + float(G6.ref_Ew(R0, (y - yh)[:n3].double(), cmt, MM))
        rs.update(E=E0, E_ref_rel=abs(E0 - Er) / max(abs(Er), 1e-300), f_ref_rel=G2._rel(f0[:n3], fr))
        gd = float((f0[:n3] * d[:n3].double()).sum())
        fd = []
        for eps in (1e-1, 1e-2):
            yp = (y + eps * d).contiguous(); ym = (y - eps * d).contiguous()
            fp, Ep = fo(yp); fmn, Em = fo(ym)
            g_fd = (Ep - Em) / (2 * eps)
            Hd_fd = (fp - fmn)[:n3] / (2 * eps)
            dd = dict(eps=eps, g_fd=g_fd, g_an=gd, g_rel=abs(g_fd - gd) / max(abs(gd), 1e-300), Hd_rel=G2._rel(mv(d)[:n3], Hd_fd))
            fd.append(dd)
        Hr = G6.ref_Gw(R0, d[:n3].double(), kt + cmt, MM, nnode)
        rs.update(fd=fd, Hd_ref_rel=G2._rel(mv(d)[:n3], Hr))
        # (C) 선탐색: SC[40+i] = ½k E(ŷ + T(η + c δ)) + ½cm E(T(η + c δ)), c = 2^-i
        nrm = torch.tensor([0.6, 0.0, 0.8], device=dev, dtype=torch.float64)
        eta = (torch.randn(capN * 3, device=dev, generator=gen) * sc).float()
        dl = (torch.randn(capN * 3, device=dev, generator=gen) * sc).float()
        sk = (torch.rand(capN, device=dev, generator=gen) * 0.7 + 0.3).float()
        pm = torch.randint(0, 3, (capN,), device=dev, generator=gen).int()

        def ls(et, i0=0):
            GQ = torch.zeros(8, device=dev, dtype=torch.float64)
            SC = torch.zeros(64, device=dev, dtype=torch.float64)
            s.k("gq_ls", capQc, (P(s.IP), P(s.cb), P(fm), P(fw), P(yh), P(None), P(et), P(dl), P(sk), P(pm), f32(0.6), f32(0.0), f32(0.8), P(GQ), P(None)), 128)
            s.k("gq_lsfin", 1, (P(SC), np.int32(10), np.int32(i0), P(None), P(GQ), f32(kt), f32(cmt)), 1)
            torch.cuda.synchronize()
            return SC[40:50].cpu().numpy()

        def Tof(v):
            return G2._T(v[:n3].double().reshape(-1, 3), sk[:nnode].double(), pm[:nnode], nrm).reshape(-1)
        got = ls(eta)
        e_ker, e_ref = [], []
        for i in (0, 3, 6, 9):
            c = 0.5 ** i
            te = Tof(eta + c * dl)
            yc = torch.zeros(capN * 3, device=dev)
            yc[:n3] = (yh[:n3].double() + te).float()
            _, Ec = fo(yc.contiguous())
            e_ker.append(abs(got[i] - Ec) / max(abs(Ec), 1e-300))
            e_ref.append(abs(got[i] - (float(G6.ref_Ew(R0, yh[:n3].double() + te, kt, MM)) + float(G6.ref_Ew(R0, te, cmt, MM)))) / max(abs(got[i]), 1e-300))
        # c = 1 에서 dE/dc: 선탐색 커널 (η ± ε δ) 의 유한차분 대 기울기 커널 · T δ
        lsfd = []
        tdl = Tof(dl)
        yc = torch.zeros(capN * 3, device=dev)
        yc[:n3] = (yh[:n3].double() + Tof(eta + dl)).float()
        fc_, _ = fo(yc.contiguous())
        gan = float((fc_[:n3] * tdl).sum())
        for eps in (1e-1, 1e-2):
            Ep = ls((eta + eps * dl).contiguous())[0]
            Em = ls((eta - eps * dl).contiguous())[0]
            gf = (Ep - Em) / (2 * eps)
            lsfd.append(dict(eps=eps, dEdc_fd=gf, dEdc_an=gan, rel=abs(gf - gan) / max(abs(gan), 1e-300)))
        rs.update(ls_vs_force_max=float(max(e_ker)), ls_vs_ref_max=float(max(e_ref)), ls_fd=lsfd)
        # Φ (옛 실행기 en_phi_n 자리): gq_phi = gq_force E
        MO = torch.ones(1, device=dev, dtype=torch.int32)
        PH = torch.zeros(2, device=dev, dtype=torch.float64)
        s.k("gq_phi", capQc, (P(s.IP), P(s.cb), P(fm), P(fw), f32(kt), f32(cmt), P(y), P(yh), P(None), P(MO), P(PH)), 128)
        torch.cuda.synchronize()
        rs["phi_rel"] = abs(float(PH[1]) - E0) / max(abs(E0), 1e-300)
        # (D) 훅 자리
        if not getattr(s, "det", False) and getattr(s, "_g3args", None):
            s._lgin = False
            rs["hook"] = G6._hook_check(s, R0, MM, nnode, gen, sc)
    finally:
        s._g3chk = False
        s._lgin = False
        s._gpin = False
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    with open(os.path.join(out, "lgfd.jsonl"), "a") as fo_:
        fo_.write(json.dumps(rs, ensure_ascii=False, default=float) + "\n")
    return rs
