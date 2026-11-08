"""절 CM73 묶음 lightg3 (11-08 위임): 가벼운 노드 튐 3 -- 절 CM71 λ_free 를 위상 기준으로 다시 정의 ((A)·(B)) + 수박 금 면 진단.

절 CM71 결과: λ_free = max_{면 36 노드, m>0, pm ≠ 2} l_n (l_n = m̄/(m_n + m̄)), γ̄_L 10 'P1X 밖만' 은 수박 튐·이탈·발산·되튐을 모두 없앴으나
  wolf 는 네 후보 모두 무너짐 -- wolf 겉층 고정 노드 면에 절 CM67 (2) (장벽 받침 표시로 고정이 풀린 가벼운 노드, m < m̄) 가 같이 껴 λ 가 0.85~0.97 로 남음.
새 정의 (문턱 없음, 위상 기준; 기존 스케일 m̄ 와 기존 고정 판정만):
  (A) AF_L3_FR=1: λ_A = max_{n ∈ 면 36 노드, m_n > 0, pm_n ≠ 2, m_n ≥ m̄} l_n  -- '원래 고정 판정 (floor_n 의 m < m̄ -> 고정) 에서 자유' 인 노드만.
      절 CM67 (2) 로 고정이 풀린 장벽 받침 노드 (m < m̄ 인데 pm ≠ 2) 를 뺌. 구속 상자 노드 (pm 2) 도 그대로 뺌. 남는 노드는 l ≤ ½.
  (B) AF_L3_FR=2: 면 36 노드 중 장벽 받침 표시 노드 (절 CM67 (2) mf_mark 가 이 걸음 표시한 노드 = 장벽 받침 안 표본점의 27 노드) 가 하나라도 있으면 그 면의 λ = 0,
      아니면 λ_A. (표시 안 된 가벼운 노드는 고정이므로 남은 면에서는 λ_A = λ_free.)
  (AF_L3_FR=0 = 절 CM71 λ_free 그대로.)
  면 가중 (ghost6 g5w): 범위 AF_L3_SC 0 '전부' fw = 1[P1X] + r λ · 1 'P1X 밖만' fw = 1[P1X] + (1 − 1[P1X]) r λ,  r = γ̄_L / γ̄.
  가중은 걸음마다 바닥 노드 커널 바로 뒤 (절 CM71 그대로), 프레임 시작 ~ 첫 바닥 노드 커널 사이는 λ 몫 0.
  m̄ = 0 인 씬 (레고·ficus) 은 아무 커널도 안 띄움 (P1X 그대로).
  AF_L3_R0=1: 진단 전용 -- 가중은 1[P1X] 만 (r 을 0 으로 씀 = 수학적으로 G''), λ 는 계산·기록 (G'' 에서 금·튐 면의 λ 를 보려고).

진단 (결과 불변): AF_L3_DG=1 -> lg3.jsonl (프레임 끝: λ_free [CM71] · λ_A · λ_B · λ_all 분포, 면 무리별 (36 노드 중 가장 가벼운 노드 m_min: 고정 후보 낌 m_min < m̄ ·
  튐 띠 m̄ ≤ m_min < 50 m̄ · 무거움; 보고용 분류) P1X 밖 λ, 표시 낀 면 몫),
  AF_L3_CK=1 -> 금 면 진단 (lg3.jsonl 의 ck): 입자 logJp > 0 (손상) 이 든 칸과 이번 프레임 새로 logJp > 0 이 된 입자가 든 칸, 그 칸에 닿는 면의 가중·벌점 에너지 몫,
  AF_L3_FD=프레임들 -> lg3fd.jsonl (그래프 밖: 가중 커널 대 torch, 유한차분, 선탐색, 훅 자리).
"""
import json
import os

import numpy as np
import torch

GL = float(os.environ.get("AF_L3_G", "0"))                                                 # γ̄_L
SC = int(os.environ.get("AF_L3_SC", "0"))                                                   # 0 전부 · 1 P1X 밖만
FR = int(os.environ.get("AF_L3_FR", "1"))                                                   # 0 절 CM71 λ_free · 1 (A) · 2 (B)
R0 = int(os.environ.get("AF_L3_R0", "0"))                                                   # 1 = 진단 전용 (λ 몫 0)
DG = int(os.environ.get("AF_L3_DG", "0"))
CK = int(os.environ.get("AF_L3_CK", "0"))
FD = {int(q) for q in os.environ.get("AF_L3_FD", "").split(",") if q.strip()}
FLOORN = ("ao_floor_n", "rt_floor_n", "mt_floor_n")
PEN = ("gq_force", "gq_mv", "gq_ls", "gq_phi")
ST = dict(on=None, log=None, out=".", nw=0, nw0=0, said=False, pre=False, npre=0, npre_tot=0, nfl=0, ljprev=None)
FRN = {0: "λ_free (절 CM71)", 1: "(A) 원래 고정 판정에서 자유", 2: "(B) (A) + 장벽 받침 표시 낀 면 뺌"}

LG_SRC = r'''
// ---- 절 CM73 lightg3: 면 가중 = 1[P1X 면] + r · λ (lib/anchorflow/fem_lightg3.py), λ 정의 fr: 0 = 절 CM71 λ_free (pm ≠ 2) · 1 = (A) 원래 고정 판정에서 자유 (pm ≠ 2, m ≥ m̄) · 2 = (B) (A) + 장벽 받침 표시 노드가 낀 면은 0
__global__ void gq_l3thk(const int* IP, const int* cb, const int* cidx, const unsigned char* cut, unsigned char* thk) {
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
// 면 (칸 c, 축 a) 36 노드: l = mbar/(m + mbar) (m > 0) 의 최댓값 -- l0 자유 (pm ≠ 2), lA 자유이면서 m ≥ mbar (원래 고정 판정 'm < mbar' 에서 자유), la 전체 (진단);
// bt = 36 노드 중 장벽 받침 표시 노드 (절 CM67 (2) mf_mark, bf) 가 있음; mi = 가장 가벼운 노드 질량 (진단 무리). pm 이 null 이면 λ 몫 0 (프레임 첫 바닥 노드 커널 앞).
// fw = ind + (sc && ind ? 0 : r · λ), λ = fr 0: l0 · 1: lA · 2: bt ? 0 : lA
__global__ void gq_l3w(const int* IP, const int* cb, const int* cidx, const unsigned char* fm, const unsigned char* thk, const float* mn, const int* pm, const unsigned char* bf,
                       float mbar, float r, int sc, int fr, float* fw, float* wl0, float* wlA, float* wbt, float* wla, float* wmi, float* wid) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int m = fm[c];
    int n[3] = {IP[6]-2, IP[7]-2, IP[8]-2};
    int s0, s1; int base = gq_base(IP, cb, c, &s0, &s1);
    float l0 = 0.f, lA0 = 0.f, a0 = 0.f, mi0 = 3.0e38f; int t0 = 0;
    if (m) {
        for (int o=0;o<27;o++) {
            int id = base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3);
            float q = mn[id];
            if (bf && bf[id]) t0 = 1;
            if (q > 0.f) { float l = mbar/(q + mbar); a0 = fmaxf(a0, l); mi0 = fminf(mi0, q);
                           if (pm && pm[id] != 2) { l0 = fmaxf(l0, l); if (q >= mbar) lA0 = fmaxf(lA0, l); } }
        }
    }
    for (int a=0;a<3;a++) {
        float w = 0.f, lx = 0.f, lA = 0.f, la = 0.f, mi = 0.f, bt = 0.f; int ind = 0;
        if (m & (1 << a)) {
            int x2[3] = {cb[c*3], cb[c*3+1], cb[c*3+2]}; x2[a] += 1;
            int c2 = (x2[a] < n[a]) ? cidx[(x2[0]*n[1] + x2[1])*n[2] + x2[2]] : -1;
            float l1 = l0, A1 = lA0, a1 = a0; int t1 = t0; mi = mi0;
            for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
                int id = gq_en(base, a, i, j, s0, s1);
                float q = mn[id];
                if (bf && bf[id]) t1 = 1;
                if (q > 0.f) { float l = mbar/(q + mbar); a1 = fmaxf(a1, l); mi = fminf(mi, q);
                               if (pm && pm[id] != 2) { l1 = fmaxf(l1, l); if (q >= mbar) A1 = fmaxf(A1, l); } }
            }
            lx = l1; lA = A1; la = a1; bt = (float)t1;
            float lam = (fr == 0) ? l1 : ((fr == 1) ? A1 : (t1 ? 0.f : A1));
            ind = (c2 >= 0 && thk[c] && thk[c2]) ? 1 : 0;
            w = (float)ind + ((sc && ind) ? 0.f : r*lam);
        }
        fw[c*3+a] = w; wl0[c*3+a] = lx; wlA[c*3+a] = lA; wbt[c*3+a] = bt; wla[c*3+a] = la; wmi[c*3+a] = mi; wid[c*3+a] = (float)ind;
    }
}
'''


def _append(src):
    if "gq_l3w(" in src:
        return src
    assert "gq_base(" in src and "gq_en(" in src, "lightg3: ghost6 소스 (gq_base·gq_en) 가 먼저 붙어야 함"
    i = src.rstrip().rfind("}")
    return src[:i] + LG_SRC + "\n}\n"


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _on(s):
    return float(getattr(s, "mthr", 0.0) or 0.0) > 0.0 and GL > 0.0


def _alloc(s):
    """버퍼 (새로 잡으면 True)."""
    cap = int(s.capQc)
    if getattr(s, "l3thk", None) is None or s.l3thk.numel() < cap:
        s.l3thk = torch.zeros(cap, device=s.dev, dtype=torch.uint8)
        s.l3l0 = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
        s.l3lA = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
        s.l3bt = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
        s.l3la = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
        s.l3mi = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
        s.l3id = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
        return True
    return False


def ratio():
    from . import fem_ghost6 as G6
    return GL / float(G6.GAMMA)


def reff():
    """가중에 실제로 쓰는 r (진단 전용 판은 0)."""
    return 0.0 if R0 else ratio()


def lam_of(s, keys):
    """이 걸음 가중에 쓴 정의의 λ (커널 출력에서)."""
    l0, lA, bt = s.l3l0[keys], s.l3lA[keys], s.l3bt[keys]
    return l0 if FR == 0 else (lA if FR == 1 else torch.where(bt > 0.5, torch.zeros_like(lA), lA))


def thick(s, ok):
    """프레임마다 rt_vn 뒤: 두꺼운 칸 (P1X 지시용)."""
    _alloc(s)
    ok(s, "gq_l3thk", int(s.capQc), (P(s.IP), P(s.cb), P(s.cidx), P(s.g3cut), P(s.l3thk)), 128)


def weights(s, ok, pm):
    """면 가중 (ghost6 g5w 에): pm = None 이면 λ 몫 0 (프레임 첫 바닥 노드 커널 앞), 아니면 그 걸음 고정 판정 (pm 2) 을 뺀 자유 노드 가벼움."""
    if _alloc(s):                                                                              # 칸 용량이 걸음 사이에 커졌으면 두꺼운 칸부터 다시
        thick(s, ok)
    cq = int(s.capQc)
    bf = getattr(s, "mfbf", None)
    assert FR != 2 or bf is not None, "lightg3 (B): 절 CM67 (2) 표시 버퍼 (s.mfbf) 없음 -- @mf2 1 이 켜져 있어야 함"
    ok(s, "gq_l3w", cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.g3fm), P(s.l3thk), P(s.mn), P(pm), P(bf), np.float32(s.mthr), np.float32(reff()), np.int32(SC), np.int32(FR),
                         P(s.g5w), P(s.l3l0), P(s.l3lA), P(s.l3bt), P(s.l3la), P(s.l3mi), P(s.l3id)), 128)
    if pm is None:
        ST["nw0"] += 1
    else:
        ST["nw"] += 1


def _npen(G6):
    return sum(int(G6.ST["nk"].get(q, 0)) for q in PEN)


def install(out):
    """실행기 runpy 순간 (다른 훅 뒤 = 가장 바깥) 에 부름."""
    from . import fem_an_rt as ART
    from . import fem_an_rt_v0 as V0
    from . import fem_ao_rt as AORT
    from . import fem_wmrx as WRX
    from . import fem_ghost6 as G6
    assert G6.ST["on"], "lightg3: ghost6 (P1X) 가 켜져 있어야 함"
    assert G6.THIN == 1 and G6.WMODE == 0 and not G6.CONS, ("lightg3: 바탕은 P1X (면 방식 1, 가중 없음, 증분 판)", G6.THIN, G6.WMODE, G6.CONS)
    ST.update(out=out)
    if DG or CK:
        ST["log"] = open(os.path.join(out, "lg3.jsonl"), "w")
    o_b, o_s = ART.build_src, AORT._src
    ART.build_src = lambda *a, **kw: _append(o_b(*a, **kw))
    AORT._src = lambda *a, **kw: _append(o_s(*a, **kw))

    def wrap_k(cls):
        _ok = cls.k

        def k(s, name, n, args, b=128):
            if name.startswith("gq_") or getattr(s, "_l3in", False):
                return _ok(s, name, n, args, b)
            if name == "rt_gauss" and ST["on"] is None:                                       # 첫 프레임: 씬 결정 (m̄ > 0 이면 후보 면 전부 + 가중, 아니면 P1X 그대로)
                ST["on"] = _on(s)
                if ST["on"]:
                    G6.THIN = 0
                    G6.WMODE = 99                                                              # g5w 포인터를 커널에 넘기게 (gq5_w 의 99 = 1; 이 모듈이 rt_vn·바닥 노드 커널 뒤 덮어씀)
                    assert getattr(s, "pm", None) is not None, "lightg3: s.pm 없음"
                print(f"[절 CM73 lightg3] 씬 결정: m̄ {float(getattr(s, 'mthr', 0.0) or 0.0):.4g}, γ̄_L {GL:g} (γ̄ {G6.GAMMA:g}, r {ratio():g}, 쓰는 r {reff():g}), 범위 {'P1X 밖만' if SC else '전부'}, 정의 {FRN[FR]} -> "
                      f"{'후보 면 전부 + 가벼움 가중' if ST['on'] else '아무것도 안 함 (P1X 그대로)'}", flush=True)
            r = _ok(s, name, n, args, b)
            if ST["on"] and (name == "rt_vn" or name in FLOORN):
                s._l3in = True
                try:
                    if name == "rt_vn":
                        thick(s, _ok)
                        weights(s, _ok, None)
                        ST["pre"] = True
                        ST["pk0"] = _npen(G6)
                    else:
                        if ST["pre"]:                                                          # 프레임 첫 바닥 노드 커널: 그 앞에 돈 벌점 커널 수 (ghost6 계수기; 0 이어야)
                            ST["npre"] += _npen(G6) - ST["pk0"]
                        weights(s, _ok, s.pm)
                        ST["pre"] = False
                        ST["nfl"] += 1
                finally:
                    s._l3in = False
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
                    if DG or CK:
                        diag(s, fi)
                    if fi in FD:
                        fdcheck(s, fi, out)
                except Exception:
                    import traceback
                    with open(os.path.join(out, "lg3_err.txt"), "a") as f_:
                        f_.write(f"f{fi}\n" + traceback.format_exc()[-3000:] + "\n")
                ST["npre_tot"] += ST["npre"]
                ST["npre"] = 0
            return res
        return w
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)
    if hasattr(ART, "split_step"):
        ART.split_step = outer(ART.split_step)
    print(f"[절 CM73] lightg3 켬 -- 면 가중 1[P1X] + (γ̄_L/γ̄) λ, 정의 {FR} {FRN[FR]}, γ̄_L {GL:g}, 범위 {SC} ({'P1X 밖만' if SC else '전부'}), 진단 전용 {R0}, 진단 {DG}, 금 면 진단 {CK}, 검사 프레임 {sorted(FD)}", flush=True)


def _q(v, qs=(0.1, 0.5, 0.9)):
    if v.numel() == 0:
        return None
    return [round(float(x), 5) for x in torch.quantile(v.float()[:1000000], torch.tensor(qs, device=v.device)).tolist()]


def _face_struct(s, G6):
    """torch 면 목록 (후보 면 = 방식 0 + 구속 노드 면 빼기): fc·fa·ids (36 노드) 와 키."""
    R = G6.ref_struct3(s, 0)
    if G6.FIXX and getattr(s, "fixd", None) is not None and R["nf"] > 0:
        keep = ~s.fixd.bool()[R["ids"].reshape(R["ids"].shape[0], -1)].any(1)
        R = dict(R)
        R.update(fc=R["fc"][keep], fa=R["fa"][keep], ids=R["ids"][keep], nf=int(keep.sum()), fkey=torch.sort(R["fc"][keep] * 3 + R["fa"][keep]).values)
    return R


def _c2(s, R):
    """면의 이웃 칸 (축 +1 쪽) 번호."""
    IP = R["IP"]
    nb = IP[6:9]
    n = [nb[0] - 2, nb[1] - 2, nb[2] - 2]
    cidx = s.cidx.reshape(-1)[:n[0] * n[1] * n[2]].long().view(*n)
    x = R["cb"][R["fc"]].clone()
    x[torch.arange(x.shape[0], device=x.device), R["fa"]] += 1
    ok = (x[:, 0] < n[0]) & (x[:, 1] < n[1]) & (x[:, 2] < n[2])
    c2 = torch.full_like(R["fc"], -1)
    xi = x[ok]
    c2[ok] = cidx[xi[:, 0], xi[:, 1], xi[:, 2]]
    return c2


def _crack(s, G6, keys, rec):
    """금 면 진단 (그래프 밖, 결과 불변): 입자 logJp > 0 (손상) 이 든 칸 · 이번 프레임 새로 > 0 이 된 입자가 든 칸, 그 칸에 닿는 면 (두 칸 중 하나) 의
    가중 (P1X 지시 · λ 몫 r λ) 과 벌점 에너지 (강성 ½ k w Dᵀ M D (y) + 고스트 질량 ½ k_M w Dᵀ M D (y − ŷ), 마지막 걸음 해) 몫."""
    from . import fem_ghost2 as G2
    lj = getattr(s, "ljp", None)
    if lj is None or getattr(s, "pcd", None) is None:
        rec["ck"] = None
        return
    N = int(lj.numel())
    x = s.x[:N] if s.x.shape[0] >= N else s.x
    fin = torch.isfinite(x).all(-1) & torch.isfinite(lj)
    cur = (lj > 0) & fin
    prev = ST["ljprev"]
    new = cur & ~prev if (prev is not None and prev.numel() == cur.numel()) else torch.zeros_like(cur)
    ST["ljprev"] = cur.clone()
    nc = int(s.IP[11])
    ci = s.cidx.reshape(-1)[s.pcd[:N].long()].long()
    okc = (ci >= 0) & (ci < nc)
    cc = torch.zeros(nc, dtype=torch.bool, device=lj.device)
    cn = torch.zeros(nc, dtype=torch.bool, device=lj.device)
    cc[ci[okc & cur]] = True
    cn[ci[okc & new]] = True
    R = _face_struct(s, G6)
    ck = dict(npart=int(fin.sum()), ncrack=int(cur.sum()), nnew=int(new.sum()), ncell_crack=int(cc.sum()), ncell_new=int(cn.sum()), nface=int(R["nf"]))
    if R["nf"]:
        c1, c2 = R["fc"], _c2(s, R)
        fcr = cc[c1] | (cc[c2.clamp(min=0)] & (c2 >= 0))
        fnw = cn[c1] | (cn[c2.clamp(min=0)] & (c2 >= 0))
        key = c1 * 3 + R["fa"]
        w = s.g5w[key].double()
        ind = s.l3id[key].double()
        lam = lam_of(s, key).double()
        wl = w - ind                                                                            # 실제 걸린 λ 몫 (진단 전용 판은 0)
        k0, kM = float(s._gpk), float(s._g3kM)
        kt = k0 if k0 > 0 else 1.0
        it2 = float(getattr(s, "_g3it2", 0.0)) or 1.0
        cmt = (kM if kM > 0 else 1.0) * it2
        MM = G2._MM(s.dev)
        nn = int(s.IP[9])
        y = s.y.reshape(-1)[:nn * 3].double()
        yh = s.yh.reshape(-1)[:nn * 3].double()
        ef = torch.zeros(R["nf"], dtype=torch.float64, device=s.dev)
        for i0 in range(0, R["nf"], 200000):
            Rs = dict(ids=R["ids"][i0:i0 + 200000])
            D1, D2 = G2.ref_D(Rs, y), G2.ref_D(Rs, y - yh)
            ef[i0:i0 + 200000] = 0.5 * kt * torch.einsum("fta,ts,fsa->f", D1, MM, D1) + 0.5 * cmt * torch.einsum("fta,ts,fsa->f", D2, MM, D2)
        out = ind < 0.5
        el, ep = ef * wl, ef * ind                                                               # λ 몫 에너지 · P1X 몫 에너지
        sm = lambda v, m: float(v[m].sum())
        for nm, sel in (("crack", fcr), ("new", fnw)):
            ck[nm] = dict(nf=int(sel.sum()), nf_out=int((sel & out).sum()), nf_out_wl=int((sel & out & (wl > 0)).sum()), nf_out_lam=int((sel & out & (lam > 0)).sum()),
                          lam_out=_q(lam[sel & out]), lam_out_mean=(float(lam[sel & out].mean()) if bool((sel & out).any()) else None),
                          wl_sum=sm(wl, sel), El=sm(el, sel), Ep=sm(ep, sel), e_out_mean=(float(ef[sel & out].mean()) if bool((sel & out).any()) else None))
        ck["all"] = dict(nf_out=int(out.sum()), nf_out_wl=int((out & (wl > 0)).sum()), nf_out_lam=int((out & (lam > 0)).sum()), lam_out_mean=(float(lam[out].mean()) if bool(out.any()) else None),
                         wl_sum=float(wl.sum()), El=float(el.sum()), Ep=float(ep.sum()), e_out_mean=(float(ef[out].mean()) if bool(out.any()) else None))
        # 새 손상 입자 가운데 λ 몫이 걸린 (또는 정의상 λ > 0 인) P1X 밖 면에 닿는 칸에 든 입자 몫
        cw = torch.zeros(nc, dtype=torch.bool, device=s.dev)
        cl = torch.zeros(nc, dtype=torch.bool, device=s.dev)
        for sel, tgt in (((wl > 0) & out, cw), ((lam > 0) & out, cl)):
            tgt[c1[sel]] = True
            q2 = c2[sel]
            tgt[q2[q2 >= 0]] = True
        pn = okc & new
        pc = okc & cur
        ck["new_in_wl_cell"] = int(cw[ci[pn]].sum()) if bool(pn.any()) else 0
        ck["new_in_lam_cell"] = int(cl[ci[pn]].sum()) if bool(pn.any()) else 0
        ck["crack_in_wl_cell"] = int(cw[ci[pc]].sum()) if bool(pc.any()) else 0
        ck["crack_in_lam_cell"] = int(cl[ci[pc]].sum()) if bool(pc.any()) else 0
        ck["ncell_wl"] = int(cw.sum())
        ck["ncell_lam"] = int(cl.sum())
    rec["ck"] = ck


@torch.no_grad()
def diag(s, fi):
    """프레임 끝 (그래프 밖): 면 수·가벼움 분포 (λ_free · λ_A · λ_B · λ_all), 면 무리 = 36 노드 중 가장 가벼운 노드 질량 m_min 으로: m_min < m̄ (고정 후보 노드 낌) ·
    m̄ ≤ m_min < 50 m̄ (질량 비 1e-3~5e-2 = 튐 노드 띠) · 그 위 (무리 경계는 보고용 분류). λ 는 마지막 바닥 노드 커널의 pm·표시로 계산한 값."""
    from . import fem_ghost6 as G6
    torch.cuda.synchronize()
    nc = int(s.IP[11])
    keys = G6.mask_keys(s.g3fm, nc)
    mt = float(s.mthr)
    nn = int(s.IP[9])
    mn = s.mn[:nn]
    pm = s.pm[:nn]
    act = mn > 0
    med = float(mn[act].median()) if bool(act.any()) else 0.0
    bf = getattr(s, "mfbf", None)
    nmk = int(bf[:nn].bool().sum()) if bf is not None else None
    rec = dict(f=fi, fr=FR, r0=R0, ncand=int(keys.numel()), mbar=mt, mbar_med=(mt / med if med > 0 else None), nw=ST["nw"], nw0=ST["nw0"], nfl=ST["nfl"], npre=ST["npre"],
               nact=int(act.sum()), nfix=int((act & (pm == 2)).sum()), nlight=int((act & (mn < mt)).sum()), nlight_free=int((act & (mn < mt) & (pm != 2)).sum()), nmark=nmk)
    if keys.numel() and DG:
        fw, l0, lA, bt, la, mi = s.g5w[keys], s.l3l0[keys], s.l3lA[keys], s.l3bt[keys] > 0.5, s.l3la[keys], s.l3mi[keys]
        lB = torch.where(bt, torch.zeros_like(lA), lA)
        lam = l0 if FR == 0 else (lA if FR == 1 else lB)
        r = reff()
        p1 = s.l3id[keys] > 0.5                                                                  # 커널이 쓴 P1X 지시
        wexp = p1.float() + r * lam * ((~p1).float() if SC else 1.0)
        rec.update(np1x=int(p1.sum()), nbt=int(bt.sum()), nbt_out=int((bt & ~p1).sum()), ind_err=float((fw - wexp).abs().max()),
                   sum_lam_out=float(lam[~p1].sum()), sum_w_out=float(fw[~p1].sum()), sum_w_in=float(fw[p1].sum()),
                   l0_out=_q(l0[~p1]), lA_out=_q(lA[~p1]), lB_out=_q(lB[~p1]), la_out=_q(la[~p1]),
                   n_l0_out_pos=int((l0[~p1] > 0).sum()), n_lA_out_pos=int((lA[~p1] > 0).sum()), n_lB_out_pos=int((lB[~p1] > 0).sum()))
        grp = {}
        for nm, sel in (("fix", mi < mt), ("band", (mi >= mt) & (mi < 50 * mt)), ("heavy", mi >= 50 * mt)):
            so = sel & ~p1
            mq = lambda v: float(v[so].mean()) if bool(so.any()) else None
            grp[nm] = dict(n=int(sel.sum()), np1x=int((sel & p1).sum()), nbt_out=int((so & bt).sum()),
                           l0_out=_q(l0[so]), lA_out=_q(lA[so]), lB_out=_q(lB[so]), la_out=_q(la[so]),
                           l0_out_mean=mq(l0), lA_out_mean=mq(lA), lB_out_mean=mq(lB), la_out_mean=mq(la),
                           l0_all=_q(l0[sel]), lA_all=_q(lA[sel]), lB_all=_q(lB[sel]))
        rec["grp"] = grp
    if CK:
        try:
            _crack(s, G6, keys, rec)
        except Exception:
            import traceback
            rec["ck_err"] = traceback.format_exc()[-1500:]
    ST["log"].write(json.dumps(rec) + "\n")
    ST["log"].flush()


def _ref(s, G6):
    """torch 참값: 후보 면 (구속 노드 면 뺀 뒤) 36 노드, P1X 지시, 정의별 λ (pm·질량·표시로 따로 계산), 가중."""
    R0_ = _face_struct(s, G6)
    R1 = G6.ref_struct3(s, 1)
    key = R0_["fc"] * 3 + R0_["fa"]
    ind = torch.isin(key, R1["fkey"]).double()
    nn = int(s.IP[9])
    mn = s.mn[:nn].double()
    pm = s.pm[:nn]
    ids = R0_["ids"].reshape(R0_["ids"].shape[0], -1)
    mm = mn[ids]
    fr = pm[ids] != 2
    mb = float(s.mthr)
    l = torch.where(mm > 0, mb / (mm + mb), torch.zeros_like(mm))
    l0 = torch.where(fr, l, torch.zeros_like(l)).max(1).values
    lA = torch.where(fr & (mm >= mb), l, torch.zeros_like(l)).max(1).values
    bf = getattr(s, "mfbf", None)
    bt = bf[:nn].bool()[ids].any(1) if bf is not None else torch.zeros_like(lA, dtype=torch.bool)
    lB = torch.where(bt, torch.zeros_like(lA), lA)
    la = l.max(1).values
    lam = l0 if FR == 0 else (lA if FR == 1 else lB)
    R0_ = dict(R0_)
    R0_["w"] = ind + reff() * lam * ((1.0 - ind) if SC else 1.0)
    return R0_, key, ind, lam, la, dict(l0=l0, lA=lA, bt=bt.double())


def fdcheck(s, fi, out):
    """프레임 fi 걸음 뒤 (그래프 밖, 결과 불변): (A) 가중 커널 대 torch, (B) 에너지 -> 기울기 유한차분, 기울기 -> 헤시안 곱 유한차분 (커널끼리) + torch 참값,
    (C) 선탐색 후보 에너지 대 같은 점의 에너지 커널 + 선탐색 에너지의 c 방향 유한차분 대 기울기, (D) 훅 자리 (실행기 커널 '훅 켬 − 훅 끔' 대 가중 참값)."""
    from . import fem_ghost6 as G6
    from . import fem_ghost2 as G2
    torch.cuda.synchronize()
    G2._skip0(s)
    s._g3chk = True
    s._l3in = True
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
        R0, key, ind, lx, la, LL = _ref(s, G6)
        kk = G6.mask_keys(fm, nc)
        rs.update(nface=int(kk.numel()), nface_ref=int(R0["nf"]), set_eq=bool(kk.numel() == R0["fkey"].numel() and bool((kk == R0["fkey"]).all())),
                  np1x=int(ind.sum()), r=reff(), k=k0, kM=kM, kt=kt, cmt=cmt)
        if R0["nf"]:
            wk = fw[key].double()
            rs.update(w_abs=float((wk - R0["w"]).abs().max()), id_abs=float((s.l3id[key].double() - ind).abs().max()), lx_abs=float((lam_of(s, key).double() - lx).abs().max()), la_abs=float((s.l3la[key].double() - la).abs().max()),
                      l0_abs=float((s.l3l0[key].double() - LL["l0"]).abs().max()), lA_abs=float((s.l3lA[key].double() - LL["lA"]).abs().max()), bt_abs=float((s.l3bt[key].double() - LL["bt"]).abs().max()),
                      nbt=int(LL["bt"].sum()), n_lA_lt_l0=int((LL["lA"] < LL["l0"]).sum()),
                      w_q=_q(R0["w"]), lx_q=_q(lx), la_q=_q(la), n_lower=int((lx < la).sum()), npm2=int((s.pm[:nnode] == 2).sum()))
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
            s._l3in = False
            rs["hook"] = G6._hook_check(s, R0, MM, nnode, gen, sc)
    finally:
        s._g3chk = False
        s._l3in = False
        s._gpin = False
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    with open(os.path.join(out, "lg3fd.jsonl"), "a") as fo_:
        fo_.write(json.dumps(rs, ensure_ascii=False, default=float) + "\n")
    return rs
