"""절 CM71 묶음 lightg2 (11-08 위임): 가벼운 노드 튐 -- 절 CM69 노드 가벼움 면 가중을 '고정되지 않은 자유 노드만' 의 가벼움으로 (λ_free).

절 CM69 결과: λ_F = max_{면 36 노드, m>0} l_n (l_n = m̄/(m_n + m̄)) 로 γ̄_L 1 은 수박 튐·이탈을 없앴으나, wolf 는 고정 노드 (m < m̄, 관성 예측에 묶임 = pm 2)
  가 낀 겉층 면 (λ ≈ 1, P1X 밖 약 4 천 개) 때문에 무너짐. 튐 노드가 낀 면의 λ 는 0.10~0.17.
새 정의 (새 문턱·클램프 없음; 기존 스케일 m̄ 와 기존 고정 판정 (floor_n 의 pm 2) 만):
  λ_F = max_{n ∈ 면 36 노드, m_n > 0, n 이 고정 아님 (pm_n ≠ 2)} l_n   (자유 노드가 없으면 0)
    고정 노드는 이미 관성 예측 ŷ 에 묶여 (미지수 아님) 튀지 않으니 벌할 이유가 없음 -> 고정 노드만 낀 겉층 면은 λ 가 그 면의 가장 가벼운 '자유' 노드 몫으로 내려감.
    고정 판정은 걸음마다 바닥 노드 커널 (ao_floor_n · rt_floor_n · mt_floor_n; 절 CM67 (2) 장벽 받침 노드 빼기 반영 뒤) 이 정하므로, 가중도 걸음마다 그 커널 바로 뒤에 다시 씀 (걸음 안에서는 상수).
  면 가중 (ghost6 g5w): 범위 AF_L2_SC 0 '전부' fw = 1[P1X] + r λ_F  (절 CM69 꼴) · 1 'P1X 밖만' fw = 1[P1X] + (1 − 1[P1X]) r λ_F,  r = γ̄_L / γ̄.
  m̄ = 0 인 씬 (레고·ficus) 은 λ ≡ 0 -> 아무 커널도 안 띄움 (P1X 그대로).
  프레임 시작 (rt_vn 뒤) 부터 그 프레임 첫 바닥 노드 커널 전까지는 pm 이 앞 프레임 것 (노드 번호가 바뀜) 이라 가중 = 1[P1X] (λ 몫 0) 으로 두고,
    그 사이에 벌점 커널 (gq_force·gq_mv·gq_ls·gq_phi) 이 몇 번 도는지 세어 진단에 남김 (0 이어야 정의대로).

진단 (결과 불변): AF_L2_DG=1 -> lg2.jsonl (프레임 끝: 후보 면·P1X 면 수, λ_free 와 절 CM69 λ_all 분포, 면 무리 (36 노드 중 가장 가벼운 노드 m_min: 고정 노드 낌 m_min < m̄ · 튐 띠 m̄ ≤ m_min < 50 m̄ · 무거움) 별 P1X 밖·전체 λ),
  AF_L2_FD=프레임들 -> lg2fd.jsonl (그래프 밖: 가중 커널 대 torch (pm 으로 따로 계산), 기울기·헤시안 곱·선탐색 에너지 유한차분, 훅 자리).
"""
import json
import os

import numpy as np
import torch

GL = float(os.environ.get("AF_L2_G", "0"))                                                 # γ̄_L
SC = int(os.environ.get("AF_L2_SC", "0"))                                                   # 0 전부 · 1 P1X 밖만
DG = int(os.environ.get("AF_L2_DG", "0"))
FD = {int(q) for q in os.environ.get("AF_L2_FD", "").split(",") if q.strip()}
FLOORN = ("ao_floor_n", "rt_floor_n", "mt_floor_n")
PEN = ("gq_force", "gq_mv", "gq_ls", "gq_phi")
ST = dict(on=None, log=None, out=".", nw=0, nw0=0, said=False, pre=False, npre=0, npre_tot=0, nfl=0)

LG_SRC = r'''
// ---- 절 CM71 lightg2: 면 가중 = 1[P1X 면] + r · (면 36 노드 중 자유 노드 (m > 0, pm ≠ 2) 가벼움 최댓값) (lib/anchorflow/fem_lightg2.py) ----
// 두꺼운 칸 (3x3x3 열림, ghost6 gq_thk thin 1 과 같은 식): 닫힌 27 이웃에 경계 아닌 활성 칸이 있음
__global__ void gq_l2thk(const int* IP, const int* cb, const int* cidx, const unsigned char* cut, unsigned char* thk) {
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
// 면 (칸 c, 축 a) 36 노드 (m > 0): 자유 노드 (pm ≠ 2; pm 이 null 이면 λ 몫 0 = 프레임 첫 걸음 앞) 가벼움 l = mbar/(m + mbar) 최댓값 lx, 전체 노드 최댓값 la (절 CM69 정의, 진단),
// 가장 가벼운 노드 질량 mi (전체, 진단 무리), fw = ind + (sc && ind ? 0 : r lx)
__global__ void gq_l2w(const int* IP, const int* cb, const int* cidx, const unsigned char* fm, const unsigned char* thk, const float* mn, const int* pm, float mbar, float r, int sc,
                       float* fw, float* wlx, float* wla, float* wmi, float* wid) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int m = fm[c];
    int n[3] = {IP[6]-2, IP[7]-2, IP[8]-2};
    int s0, s1; int base = gq_base(IP, cb, c, &s0, &s1);
    float l0 = 0.f, a0 = 0.f, mi0 = 3.0e38f;
    if (m) {
        for (int o=0;o<27;o++) {
            int id = base + (o/9)*s0 + ((o/3)%3)*s1 + (o%3);
            float q = mn[id];
            if (q > 0.f) { float l = mbar/(q + mbar); a0 = fmaxf(a0, l); mi0 = fminf(mi0, q); if (pm && pm[id] != 2) l0 = fmaxf(l0, l); }
        }
    }
    for (int a=0;a<3;a++) {
        float w = 0.f, lx = 0.f, la = 0.f, mi = 0.f; int ind = 0;
        if (m & (1 << a)) {
            int x2[3] = {cb[c*3], cb[c*3+1], cb[c*3+2]}; x2[a] += 1;
            int c2 = (x2[a] < n[a]) ? cidx[(x2[0]*n[1] + x2[1])*n[2] + x2[2]] : -1;
            float l1 = l0, a1 = a0; mi = mi0;
            for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
                int id = gq_en(base, a, i, j, s0, s1);
                float q = mn[id];
                if (q > 0.f) { float l = mbar/(q + mbar); a1 = fmaxf(a1, l); mi = fminf(mi, q); if (pm && pm[id] != 2) l1 = fmaxf(l1, l); }
            }
            lx = l1; la = a1;
            ind = (c2 >= 0 && thk[c] && thk[c2]) ? 1 : 0;
            w = (float)ind + ((sc && ind) ? 0.f : r*lx);
        }
        fw[c*3+a] = w; wlx[c*3+a] = lx; wla[c*3+a] = la; wmi[c*3+a] = mi; wid[c*3+a] = (float)ind;
    }
}
'''


def _append(src):
    if "gq_l2w(" in src:
        return src
    assert "gq_base(" in src and "gq_en(" in src, "lightg2: ghost6 소스 (gq_base·gq_en) 가 먼저 붙어야 함"
    i = src.rstrip().rfind("}")
    return src[:i] + LG_SRC + "\n}\n"


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _on(s):
    return float(getattr(s, "mthr", 0.0) or 0.0) > 0.0 and GL > 0.0


def _alloc(s):
    """버퍼 (새로 잡으면 True)."""
    cap = int(s.capQc)
    if getattr(s, "l2thk", None) is None or s.l2thk.numel() < cap:
        s.l2thk = torch.zeros(cap, device=s.dev, dtype=torch.uint8)
        s.l2lx = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
        s.l2la = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
        s.l2mi = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
        s.l2id = torch.zeros(cap * 3, device=s.dev, dtype=torch.float32)
        return True
    return False


def ratio():
    from . import fem_ghost6 as G6
    return GL / float(G6.GAMMA)


def thick(s, ok):
    """프레임마다 rt_vn 뒤: 두꺼운 칸 (P1X 지시용)."""
    _alloc(s)
    ok(s, "gq_l2thk", int(s.capQc), (P(s.IP), P(s.cb), P(s.cidx), P(s.g3cut), P(s.l2thk)), 128)


def weights(s, ok, pm):
    """면 가중 (ghost6 g5w 에): pm = None 이면 λ 몫 0 (프레임 첫 바닥 노드 커널 앞), 아니면 그 걸음 고정 판정 (pm 2) 을 뺀 자유 노드 가벼움."""
    if _alloc(s):                                                                              # 칸 용량이 걸음 사이에 커졌으면 두꺼운 칸부터 다시
        thick(s, ok)
    cq = int(s.capQc)
    ok(s, "gq_l2w", cq, (P(s.IP), P(s.cb), P(s.cidx), P(s.g3fm), P(s.l2thk), P(s.mn), P(pm), np.float32(s.mthr), np.float32(ratio()), np.int32(SC),
                         P(s.g5w), P(s.l2lx), P(s.l2la), P(s.l2mi), P(s.l2id)), 128)
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
    assert G6.ST["on"], "lightg2: ghost6 (P1X) 가 켜져 있어야 함"
    assert G6.THIN == 1 and G6.WMODE == 0 and not G6.CONS, ("lightg2: 바탕은 P1X (면 방식 1, 가중 없음, 증분 판)", G6.THIN, G6.WMODE, G6.CONS)
    ST.update(out=out)
    if DG:
        ST["log"] = open(os.path.join(out, "lg2.jsonl"), "w")
    o_b, o_s = ART.build_src, AORT._src
    ART.build_src = lambda *a, **kw: _append(o_b(*a, **kw))
    AORT._src = lambda *a, **kw: _append(o_s(*a, **kw))

    def wrap_k(cls):
        _ok = cls.k

        def k(s, name, n, args, b=128):
            if name.startswith("gq_") or getattr(s, "_l2in", False):
                return _ok(s, name, n, args, b)
            if name == "rt_gauss" and ST["on"] is None:                                       # 첫 프레임: 씬 결정 (m̄ > 0 이면 후보 면 전부 + 가중, 아니면 P1X 그대로)
                ST["on"] = _on(s)
                if ST["on"]:
                    G6.THIN = 0
                    G6.WMODE = 99                                                              # g5w 포인터를 커널에 넘기게 (gq5_w 의 99 = 1; 이 모듈이 rt_vn·바닥 노드 커널 뒤 덮어씀)
                    assert getattr(s, "pm", None) is not None, "lightg2: s.pm 없음"
                print(f"[절 CM71 lightg2] 씬 결정: m̄ {float(getattr(s, 'mthr', 0.0) or 0.0):.4g}, γ̄_L {GL:g} (γ̄ {G6.GAMMA:g}, r {ratio():g}), 범위 {'P1X 밖만' if SC else '전부'} -> "
                      f"{'후보 면 전부 + 자유 노드 가벼움 가중' if ST['on'] else '아무것도 안 함 (P1X 그대로)'}", flush=True)
            r = _ok(s, name, n, args, b)
            if ST["on"] and (name == "rt_vn" or name in FLOORN):
                s._l2in = True
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
                    s._l2in = False
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
                    with open(os.path.join(out, "lg2_err.txt"), "a") as f_:
                        f_.write(f"f{fi}\n" + traceback.format_exc()[-3000:] + "\n")
                ST["npre_tot"] += ST["npre"]
                ST["npre"] = 0
            return res
        return w
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)
    if hasattr(ART, "split_step"):
        ART.split_step = outer(ART.split_step)
    print(f"[절 CM71] lightg2 켬 -- 면 가중 1[P1X] + (γ̄_L/γ̄) max 자유 노드 (pm ≠ 2) 가벼움 m̄/(m+m̄), γ̄_L {GL:g}, 범위 {SC} ({'P1X 밖만' if SC else '전부'}), 진단 {DG}, 검사 프레임 {sorted(FD)}", flush=True)


def _q(v, qs=(0.1, 0.5, 0.9)):
    if v.numel() == 0:
        return None
    return [round(float(x), 5) for x in torch.quantile(v.float()[:1000000], torch.tensor(qs, device=v.device)).tolist()]


@torch.no_grad()
def diag(s, fi):
    """프레임 끝 (그래프 밖): 면 수·가벼움 분포. 면 무리 = 그 면 36 노드 중 가장 가벼운 노드 질량 m_min 으로: m_min < m̄ (고정 후보 노드 낌) · m̄ ≤ m_min < 50 m̄ (질량 비 1e-3~5e-2 = 튐 노드 띠) · 그 위.
    (무리 경계는 보고용 분류일 뿐, 정의에 안 들어감.) λ_free = 이 걸음 가중에 쓴 값 (마지막 바닥 노드 커널의 pm), λ_all = 절 CM69 정의 (비교)."""
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
    rec = dict(f=fi, ncand=int(keys.numel()), mbar=mt, mbar_med=(mt / med if med > 0 else None), nw=ST["nw"], nw0=ST["nw0"], nfl=ST["nfl"], npre=ST["npre"],
               nact=int(act.sum()), nfix=int((act & (pm == 2)).sum()), nlight=int((act & (mn < mt)).sum()), nlight_free=int((act & (mn < mt) & (pm != 2)).sum()))
    if keys.numel():
        fw, lx, la, mi = s.g5w[keys], s.l2lx[keys], s.l2la[keys], s.l2mi[keys]
        r = ratio()
        p1 = s.l2id[keys] > 0.5                                                                  # 커널이 쓴 P1X 지시
        wexp = p1.float() + r * lx * ((~p1).float() if SC else 1.0)
        ind_err = float((fw - wexp).abs().max())
        rec.update(np1x=int(p1.sum()), sum_l=float(lx.sum()), sum_l_out=float(lx[~p1].sum()), sum_w_out=float(fw[~p1].sum()), sum_w_in=float(fw[p1].sum()), ind_err=ind_err,
                   lx_np1x=_q(lx[~p1]), la_np1x=_q(la[~p1]), lx_p1x=_q(lx[p1]), la_p1x=_q(la[p1]), n_zero_out=int((lx[~p1] == 0).sum()))
        grp = {}
        for nm, sel in (("fix", mi < mt), ("band", (mi >= mt) & (mi < 50 * mt)), ("heavy", mi >= 50 * mt)):
            so = sel & ~p1
            grp[nm] = dict(n=int(sel.sum()), np1x=int((sel & p1).sum()), lx=_q(lx[sel]), la=_q(la[sel]), lx_out=_q(lx[so]), la_out=_q(la[so]),
                           lx_mean=(float(lx[sel].mean()) if bool(sel.any()) else None), lx_out_mean=(float(lx[so].mean()) if bool(so.any()) else None),
                           la_out_mean=(float(la[so].mean()) if bool(so.any()) else None), n_out_zero=int((lx[so] == 0).sum()))
        rec["grp"] = grp
    ST["log"].write(json.dumps(rec) + "\n")
    ST["log"].flush()


def _ref(s, G6):
    """torch 참값: 후보 면 (구속 노드 면 뺀 뒤) 36 노드, P1X 지시, 자유 노드 (pm ≠ 2) 가벼움 최댓값 · 전체 노드 최댓값, 가중."""
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
    pm = s.pm[:nn]
    ids = R0["ids"].reshape(R0["ids"].shape[0], -1)
    mm = mn[ids]
    fr = pm[ids] != 2
    mb = float(s.mthr)
    l = torch.where(mm > 0, mb / (mm + mb), torch.zeros_like(mm))
    lx = torch.where(fr, l, torch.zeros_like(l)).max(1).values
    la = l.max(1).values
    R0 = dict(R0)
    R0["w"] = ind + ratio() * lx * ((1.0 - ind) if SC else 1.0)
    return R0, key, ind, lx, la


def fdcheck(s, fi, out):
    """프레임 fi 걸음 뒤 (그래프 밖, 결과 불변): (A) 가중 커널 대 torch, (B) 에너지 -> 기울기 유한차분, 기울기 -> 헤시안 곱 유한차분 (커널끼리) + torch 참값,
    (C) 선탐색 후보 에너지 대 같은 점의 에너지 커널 + 선탐색 에너지의 c 방향 유한차분 대 기울기, (D) 훅 자리 (실행기 커널 '훅 켬 − 훅 끔' 대 가중 참값)."""
    from . import fem_ghost6 as G6
    from . import fem_ghost2 as G2
    torch.cuda.synchronize()
    G2._skip0(s)
    s._g3chk = True
    s._l2in = True
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
        R0, key, ind, lx, la = _ref(s, G6)
        kk = G6.mask_keys(fm, nc)
        rs.update(nface=int(kk.numel()), nface_ref=int(R0["nf"]), set_eq=bool(kk.numel() == R0["fkey"].numel() and bool((kk == R0["fkey"]).all())),
                  np1x=int(ind.sum()), r=ratio(), k=k0, kM=kM, kt=kt, cmt=cmt)
        if R0["nf"]:
            wk = fw[key].double()
            rs.update(w_abs=float((wk - R0["w"]).abs().max()), id_abs=float((s.l2id[key].double() - ind).abs().max()), lx_abs=float((s.l2lx[key].double() - lx).abs().max()), la_abs=float((s.l2la[key].double() - la).abs().max()),
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
            s._l2in = False
            rs["hook"] = G6._hook_check(s, R0, MM, nnode, gen, sc)
    finally:
        s._g3chk = False
        s._l2in = False
        s._gpin = False
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    with open(os.path.join(out, "lg2fd.jsonl"), "a") as fo_:
        fo_.write(json.dumps(rs, ensure_ascii=False, default=float) + "\n")
    return rs
