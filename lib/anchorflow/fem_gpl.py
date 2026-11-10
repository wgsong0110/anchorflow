"""절 CM100 묶음 gpl (사용자 11-10 지시): ghost penalty 를 켠 판 (H5) 에서 P1X 강성·고스트 질량 벌점의 면 세기에 그 면 노드의 가벼움 λ 를 곱함.

벌점의 목적은 수박 겉면의 아주 가벼운 노드가 날아가는 것을 막는 것뿐인데, H5 의 P1X 벌점은 그 목적과 무관하게 wolf 모래 겉층 면 전체를 묶는다 (절 CM87·CM95·CM99).
정의 (P1X 면 집합 그대로, 새 문턱·클램프 없음):
  λ_F = λ_A (절 CM73 (A) 그대로) = max_{n ∈ 면 36 노드, m_n > 0, pm_n ≠ 2, m_n ≥ m̄} l_n,  l_n = m̄ / (m_n + m̄)  (m̄ = mlow × 첫 프레임 노드 질량 중앙, 절 CM69)
    -- 원래 고정 판정에서 자유인 노드만 (고정 노드 pm 2 와 장벽 받침으로 풀린 m < m̄ 노드는 뺌), 자유 노드가 없으면 0.
  면 가중 fw = 1[P1X] · q · λ_F + (1 − 1[P1X]) · r κ λ_F     (q = γ̄_P/γ̄, γ̄ = 0.1; 둘째 항 = H5 가벼움 벌점 몫 그대로)
    E = ½ Σ_F fw_F [ k D_F(y)·M D_F(y) + (k_M/τ²) D_F(y − ŷ)·M D_F(y − ŷ) ],  k = γ̄ (2μ+λ) h, k_M = γ̄ ρ̄ h³ (절 CM53)
  곧 P1X 면의 실제 계수 γ̄_P λ_F: 무거운 노드뿐인 면 (m ≫ m̄) 은 λ ≈ m̄/m → 0, 가벼운 자유 노드 (m ≈ m̄) 가 낀 면은 λ → ½.
  모든 씬 같은 정의: m̄ = 0 인 씬 (레고·ficus, --mlow 없음) 은 l ≡ 0 이라 P1X 벌점도 0 (정의상). wolf (κ 0) 는 둘째 항 0, P1X 몫만.
구현 (H5 사슬 위; 커널은 lightg3 의 gq_l3w 그대로 + 덧쓰기 커널 하나):
  - lightg3 씬 결정 (_on) 을 모든 씬 '켬' 으로 (H5 에서 꺼진 wolf (κ 0)·레고·ficus (m̄ 0) 도 lightg3 경로: ghost6 면 방식 0 + 가중 버퍼 g5w).
    m̄ 없는 실행기 (레고·ficus 탄성 RTSolver) 는 s.mthr = 0 을 둠 (gq_l3w 의 l = 0/(m + 0) = 0).
  - lightg3 가 g5w 를 쓸 때마다 (rt_vn 뒤 = λ 몫 0 · 바닥 노드 커널 뒤 = 그 걸음 고정 판정) 바로 뒤 gq_gpl: P1X 면 (lightg3 지시 l3id) 의 fw ← q · λ_A (l3lA).
    κ (lightg4 reff = r κ) 는 둘째 항에만 그대로 -- wolf 의 P1X 밖 면은 0.
진단 (AF_GPL_DG=1, 결과 불변): gpl.jsonl 프레임 끝 = 후보·P1X 면 수, P1X 면 λ 분위·평균, P1X 면 실효 가중 합 / P1X 면 수 (= H5 대비 P1X 벌점 몫), 면 무리 (36 노드 중
  가장 가벼운 노드 m_min: < m̄ · m̄~50 m̄ · 무거움; 보고용 분류) 별 P1X 면 수·λ 평균, 가중 대 커널 출력 식 최대 차 (werr), 앞 8 프레임과 10 프레임마다 torch 참값
  (lightg3 _ref: pm·질량에서 따로 계산한 λ_A·P1X 지시) 대 실행 가중 최대 차 (rerr). gplinfo.json = 결정·q·κ·m̄·커널 호출 수.
"""
import json
import os

import numpy as np
import torch

Q = float(os.environ.get("AF_GPL_Q", "0"))
DG = int(os.environ.get("AF_GPL_DG", "0"))
ST = dict(q=Q, dec=None, kap=None, mbar=None, l3base=None, nmul=0, nmul0=0, nfr=0, nref=0, rerr_max=None, werr_max=None, out=".", log=None)

GPL_SRC = r'''
// ---- 절 CM100 gpl: P1X 면 가중 = q · λ_A (lightg3 가 쓴 가중의 P1X 몫 덮어씀; lib/anchorflow/fem_gpl.py)
__global__ void gq_gpl(const int* IP, const unsigned char* fm, const float* ind, const float* lA, float q, float* fw) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int m = fm[c];
    if (!m) return;
    for (int a=0;a<3;a++) {
        if (!(m & (1 << a))) continue;
        if (ind[c*3+a] > 0.5f) fw[c*3+a] = q*lA[c*3+a];
    }
}
'''


def _append(src):
    if "gq_gpl(" in src:
        return src
    i = src.rstrip().rfind("}")
    return src[:i] + GPL_SRC + "\n}\n"


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _info():
    with open(os.path.join(ST["out"], "gplinfo.json"), "w") as f_:
        json.dump({k: v for k, v in ST.items() if k not in ("log",)}, f_, ensure_ascii=False, default=str)


def _q(v):
    if v.numel() == 0:
        return None
    return [round(float(x), 5) for x in torch.quantile(v.float()[:1000000], torch.tensor([0.1, 0.5, 0.9], device=v.device)).tolist()]


@torch.no_grad()
def diag(s, fi):
    from . import fem_ghost6 as G6
    from . import fem_lightg3 as L3
    torch.cuda.synchronize()
    nc = int(s.IP[11].item())
    keys = G6.mask_keys(s.g3fm, nc)
    w = s.g5w[keys].double()
    ind = s.l3id[keys] > 0.5
    lA = s.l3lA[keys].double()
    mi = s.l3mi[keys].double()
    r = float(L3.reff())
    wexp = torch.where(ind, Q * lA, r * lA)
    mt = float(getattr(s, "mthr", 0.0) or 0.0)
    rec = dict(f=fi, ncand=int(keys.numel()), np1x=int(ind.sum()), r=r, q=Q, mbar=mt,
               werr=float((w - wexp).abs().max()) if w.numel() else 0.0,
               lam_p1x=_q(lA[ind]), lam_p1x_mean=(float(lA[ind].mean()) if bool(ind.any()) else None),
               p1x_eff=(float(w[ind].sum()) / max(int(ind.sum()), 1)),                       # P1X 면 실효 가중 평균 (H5 = 1)
               p1x_eff_rel=(float(w[ind].sum()) / Q / max(int(ind.sum()), 1)) if Q > 0 else None,  # = P1X 면 λ 평균
               lam0_frac=(float((lA[ind] < 1e-2).double().mean()) if bool(ind.any()) else None),
               out_sum=float(w[~ind].sum()), out_n=int((~ind).sum()))
    grp = {}
    for nm, sel in (("fix", mi < mt), ("band", (mi >= mt) & (mi < 50 * mt)), ("heavy", mi >= 50 * mt)):
        so = sel & ind
        grp[nm] = dict(n=int(so.sum()), lam_mean=(float(lA[so].mean()) if bool(so.any()) else None), lam=_q(lA[so]), w_sum=float(w[so].sum()))
    rec["grp"] = grp
    if ST["nfr"] <= 8 or ST["nfr"] % 10 == 0:                                                 # torch 참값 (pm·질량에서 따로; 비쌈 -> 일부 프레임)
        try:
            R0, key, indr, lam, la, LL = L3._ref(s, G6)
            if R0["nf"]:
                wr = torch.where(indr > 0.5, Q * LL["lA"], r * LL["lA"])
                rec.update(rerr=float((s.g5w[key].double() - wr).abs().max()), set_eq=bool(key.numel() == keys.numel() and bool((torch.sort(key).values == keys).all())),
                           ind_err=float((s.l3id[key].double() - indr).abs().max()), lA_err=float((s.l3lA[key].double() - LL["lA"]).abs().max()))
                ST["nref"] += 1
                ST["rerr_max"] = max(ST["rerr_max"] or 0.0, rec["rerr"])
        except Exception:
            import traceback
            rec["ref_err"] = traceback.format_exc()[-1500:]
    ST["werr_max"] = max(ST["werr_max"] or 0.0, rec["werr"])
    ST["log"].write(json.dumps(rec) + "\n")
    ST["log"].flush()


def install(out, q=None):
    """실행기 runpy 순간, lightg3·lightg4 설치 뒤 (가장 바깥 감싸개) 에 부름."""
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    from . import fem_wmrx as WRX
    from . import fem_ghost6 as G6
    from . import fem_lightg3 as L3
    from . import fem_lightg4 as L4
    global Q
    if q is not None:
        Q = float(q)
        ST["q"] = Q
    assert Q > 0.0, ("gpl: q = γ̄_P/γ̄ > 0 이어야", Q)
    assert G6.ST["on"] and G6.THIN == 1 and G6.WMODE == 0, ("gpl: 바탕은 P1X (ghost6 면 방식 1, 가중 없음)", G6.ST["on"], G6.THIN, G6.WMODE)
    assert L3.GL > 0.0 and L3.FR == 1 and L3.SC == 1 and not L3.R0, ("gpl: H5 사슬 (lightg3 (A) γ̄_L, P1X 밖만) 이어야", L3.GL, L3.FR, L3.SC, L3.R0)
    ST["out"] = out
    if DG:
        ST["log"] = open(os.path.join(out, "gpl.jsonl"), "w")
    o_b, o_s = ART.build_src, AORT._src
    ART.build_src = lambda *a, **kw: _append(o_b(*a, **kw))
    AORT._src = lambda *a, **kw: _append(o_s(*a, **kw))
    o_on, o_w = L3._on, L3.weights

    def _on(s):
        base = bool(o_on(s))                                                                    # H5 결정 (lightg3 m̄ > 0 · lightg4 κ > 0) -- 기록용
        if getattr(s, "mthr", None) is None:
            s.mthr = 0.0                                                                       # 탄성 실행기 (레고·ficus): m̄ 없음 = 0 -> l ≡ 0
        k = L4.ST["kap"] if L4.ST.get("kap") is not None else float(L4.kappa_of(s)[0])
        ST.update(dec=True, kap=float(k), mbar=float(s.mthr), l3base=base, r_out=float(L3.reff()))
        assert getattr(s, "pm", None) is not None, "gpl: s.pm 없음"
        print(f"[절 CM100 gpl] 씬 결정: m̄ {float(s.mthr):.4g}, κ {k:g}, H5 가벼움 벌점 {base} -> 모든 씬 lightg3 경로 켬: P1X 면 가중 q λ_A (q {Q:g} = γ̄_P {0.1 * Q:g}/γ̄ 0.1), "
              f"P1X 밖 면 r κ λ_A (r κ {float(L3.reff()):g}){' -- m̄ 0 이라 λ ≡ 0 (P1X 벌점 0)' if float(s.mthr) == 0.0 else ''}", flush=True)
        _info()
        return True

    def weights(s, ok, pm):
        o_w(s, ok, pm)
        ok(s, "gq_gpl", int(s.capQc), (P(s.IP), P(s.g3fm), P(s.l3id), P(s.l3lA), np.float32(Q), P(s.g5w)), 128)
        if pm is None:
            ST["nmul0"] += 1
        else:
            ST["nmul"] += 1

    L3._on = _on
    L3.weights = weights

    dep = dict(d=0)

    def outer(fn):
        def w(s, t, h, *a, **kw):
            dep["d"] += 1
            try:
                res = fn(s, t, h, *a, **kw)
            finally:
                dep["d"] -= 1
            if dep["d"] == 0 and ST["dec"]:
                fi = int(round(t / h)) + 1
                ST["nfr"] += 1
                try:
                    if DG and getattr(s, "g3fm", None) is not None and getattr(s, "l3lA", None) is not None:
                        diag(s, fi)
                    ST["npre_tot"] = int(L3.ST.get("npre_tot") or 0) + int(L3.ST.get("npre") or 0)   # 프레임 시작 ~ 첫 바닥 노드 커널 사이 벌점 커널 수 (0 이어야)
                    if ST["nfr"] <= 8 or ST["nfr"] % 10 == 0:
                        _info()
                except Exception:
                    import traceback
                    with open(os.path.join(out, "gpl_err.txt"), "a") as f_:
                        f_.write(f"f{fi}\n" + traceback.format_exc()[-3000:] + "\n")
            return res
        return w
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)
    if hasattr(ART, "split_step"):
        ART.split_step = outer(ART.split_step)
    _info()
    print(f"[절 CM100] gpl 켬 -- P1X 면 가중 q λ_A (q {Q:g}, 강성·고스트 질량), λ = 절 CM73 (A), 진단 {DG}", flush=True)
