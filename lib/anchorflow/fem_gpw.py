"""절 CM95 묶음 gpw (사용자 11-10 지시): ghost penalty 를 '거의 빈 칸에 맞닿은 면' 에만 -- P1X 면 가중 w_F = 1 − φ_min.

정의 (새 문턱·클램프 없음):
  칸 점유 φ_c = min(1, V_c / h³) -- 절 CM55 ghost5 정의 그대로 (V_c = 그 칸 가우스 점 부피 합 = 입자 부피 합, h³ = |det A|; min(1, ·) 은 몫의 정의, 새 것 아님).
  면 F = (칸 a, 칸 b) 의 가중 w_F = 1 − min(φ_a, φ_b)   (덜 찬 칸의 결손 몫 = 그 칸의 '빈' 부피 몫).
  P1X 면 집합 (절 CM53 `@g4 1e-1 @g4t 1 @g4x 1`) 위에서 강성 벌점 k 와 고스트 질량 k_M 모두에 w_F 를 곱함:
    E = ½ Σ_{F ∈ P1X} w_F [ k D_F(y)·M D_F(y) + (k_M/τ²) D_F(y − ŷ)·M D_F(y − ŷ) ].
  근거 (문헌): 가중 ghost penalty -- Frei, Knoke, Steinbach, Wenske, Wick 2024 (arXiv:2402.00209, 2.3 절) 는 면 벌점을 그 면에 닿는 칸의 잘린 부피 몫 κ_K 로
    가중 (Σ_{K ∋ F} ½ w_max^{1−2κ_K}: 거의 빈 칸 쪽 κ → 0 이면 크고 꽉 찬 칸 κ → 1 이면 작음). 벌점의 원래 목적은 잘린 칸 (작은 받침) 의 조건수 회복
    (Burman 2010, Burman & Hansbo 2012) 이라 꽉 찬 칸끼리 면은 벌할 까닭이 없음. 여기서는 그 단조 성질만 쓰고 w_max 같은 값을 안 두려고
    덜 찬 칸 결손 몫 1 − φ_min (절 CM55 방식 5 와 같은 식) 을 씀: φ_min → 0 이면 w → 1 (P1X 그대로), 두 칸 모두 꽉 차면 w → 0 (흐르는 모래 안쪽 면 안 묶음).
모드 (AF_GPW): 1 = (A) 모든 씬, 2 = (B) 재질 응집 몫 κ = 0 (절 CM75 lightg4.kappa_of; wolf) 인 씬만. 그 밖 씬 (B) 은 아무것도 안 함 (H5 그대로).
구현 (H5 사슬 위, 커널은 ghost6 것 그대로 -- 면 가중 버퍼 g5w 를 모든 벌점 커널 (gq_force·gq_mv·gq_ls·gq_phi) 이 이미 읽음):
  - lightg3 (가벼움 벌점) 이 꺼진 씬 (wolf κ 0 · 레고·ficus m̄ 0): 첫 rt_gauss 결정 때 ghost6 WMODE = 5 -> 프레임마다 rt_gauss 뒤 gq5_w 가 P1X 면에 1 − φ_min.
  - lightg3 가 켜진 씬 (수박·bread, H5 에서 g5w = 1[P1X] + (1 − 1[P1X]) r κ λ_A): lightg3 가 g5w 를 쓸 때마다 (rt_vn 뒤·바닥 노드 커널 뒤) 바로 뒤에
    gq_gpw 커널로 P1X 면 몫만 w_F 배: fw ← fw − 1[P1X] φ_min (= 1[P1X] (1 − φ_min) + 나머지 그대로).
진단 (AF_GPW_DG=1, 결과 불변, 프레임 끝): gpw.jsonl = P1X 면 수·w_F 평균·분위 (10/50/90 %)·w < ½ 몫·φ_min 중앙. gpwinfo.json = 결정·κ·커널 호출 수.
"""
import json
import os

import numpy as np
import torch

MODE = int(os.environ.get("AF_GPW", "0"))
DG = int(os.environ.get("AF_GPW_DG", "0"))
ST = dict(mode=MODE, dec=None, kap=None, path=None, nmul=0, nfr=0, log=None, out=".")

GPW_SRC = r'''
// ---- 절 CM95 gpw: lightg3 가 쓴 면 가중의 P1X 몫에 w_F = 1 − φ_min (fw ← fw − ind φ_min; lib/anchorflow/fem_gpw.py)
__global__ void gq_gpw(const int* IP, const int* cb, const int* cidx, const unsigned char* fm, const float* phi, const float* ind, float* fw) {
#ifdef AF_ADAPT
    if (AF_SKIP) return;
#endif
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (IP[13] || c >= IP[11]) return;
    int m = fm[c];
    if (!m) return;
    int n[3] = {IP[6]-2, IP[7]-2, IP[8]-2};
    for (int a=0;a<3;a++) {
        if (!(m & (1 << a))) continue;
        float id = ind[c*3+a];
        if (id <= 0.f) continue;
        int x[3] = {cb[c*3], cb[c*3+1], cb[c*3+2]}; x[a] += 1;
        int c2 = (x[a] < n[a]) ? cidx[(x[0]*n[1] + x[1])*n[2] + x[2]] : -1;
        float pa = phi[c], pb = (c2 >= 0) ? phi[c2] : 0.f;
        fw[c*3+a] -= id*fminf(pa, pb);
    }
}
'''


def _append(src):
    if "gq_gpw(" in src:
        return src
    i = src.rstrip().rfind("}")
    return src[:i] + GPW_SRC + "\n}\n"


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _info():
    with open(os.path.join(ST["out"], "gpwinfo.json"), "w") as f_:
        json.dump({k: v for k, v in ST.items() if k not in ("log",)}, f_, ensure_ascii=False, default=str)


def _q(v):
    if v.numel() == 0:
        return None
    return [round(float(x), 4) for x in torch.quantile(v.float()[:1000000], torch.tensor([0.1, 0.5, 0.9], device=v.device)).tolist()]


def diag(s, fi):
    from . import fem_ghost6 as G6
    nc = int(s.IP[11].item())
    keys = G6.mask_keys(s.g3fm, nc)
    if ST["path"] == "mul":
        keys = keys[s.l3id[keys] > 0.5]
    w = s.g5w[keys].double()
    c = keys // 3
    a = keys - 3 * c
    cb = s.cb.reshape(-1)[:nc * 3].long().view(nc, 3)
    IP = s.IP
    n0, n1, n2 = int(IP[6]) - 2, int(IP[7]) - 2, int(IP[8]) - 2
    x2 = cb[c].clone()
    x2[torch.arange(c.numel(), device=c.device), a] += 1
    cidx = s.cidx.reshape(-1).long()
    inb = (x2[:, 0] < n0) & (x2[:, 1] < n1) & (x2[:, 2] < n2)
    x2 = torch.where(inb[:, None], x2, torch.zeros_like(x2))
    c2 = torch.where(inb, cidx[(x2[:, 0] * n1 + x2[:, 1]) * n2 + x2[:, 2]], torch.full_like(c, -1))
    ph = s.g5phi
    pmin = torch.minimum(ph[c], ph[c2.clamp(min=0)])
    pmin = torch.where(c2 >= 0, pmin, torch.zeros_like(pmin))
    werr = float((w - (1.0 - pmin.double())).abs().max()) if w.numel() else 0.0
    rec = dict(f=fi, nface=int(w.numel()), wmean=round(float(w.mean()), 5) if w.numel() else None, wq=_q(w),
               whalf=round(float((w < 0.5).double().mean()), 4) if w.numel() else None, pminq=_q(pmin), werr=werr)
    ST["log"].write(json.dumps(rec) + "\n")
    ST["log"].flush()


def install(out, mode=None):
    """실행기 runpy 순간, lightg3·lightg4 설치 뒤 (가장 바깥 감싸개) 에 부름."""
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    from . import fem_wmrx as WRX
    from . import fem_ghost6 as G6
    from . import fem_lightg3 as L3
    from . import fem_lightg4 as L4
    if mode is not None:
        ST["mode"] = int(mode)
    assert ST["mode"] in (1, 2), ("gpw: 모드는 1 (A 모든 씬) · 2 (B κ 0 씬만)", ST["mode"])
    assert G6.ST["on"] and G6.THIN == 1 and G6.WMODE == 0, ("gpw: 바탕은 P1X (ghost6 면 방식 1, 가중 없음)", G6.ST["on"], G6.THIN, G6.WMODE)
    assert L3.GL > 0.0, "gpw: H5 사슬 (lightg3 --l3 γ̄_L) 이 있어야 함 (씬 결정이 lightg3 첫 rt_gauss 결정에 붙음)"
    ST["out"] = out
    if DG:
        ST["log"] = open(os.path.join(out, "gpw.jsonl"), "w")
    o_b, o_s = ART.build_src, AORT._src
    ART.build_src = lambda *a, **kw: _append(o_b(*a, **kw))
    AORT._src = lambda *a, **kw: _append(o_s(*a, **kw))
    o_on, o_w = L3._on, L3.weights

    def _on(s):
        on = bool(o_on(s))                                                                     # lightg3 (+ lightg4 κ) 결정 그대로
        k = L4.ST["kap"] if L4.ST.get("kap") is not None else float(L4.kappa_of(s)[0])
        app = ST["mode"] == 1 or (ST["mode"] == 2 and k == 0.0)
        ST.update(kap=float(k), dec=bool(app), path=("mul" if on else "wmode5") if app else "none", l3on=on)
        if app and not on:
            G6.WMODE = 5                                                                       # ghost6 gq5_w 방식 5 = 1 − φ_min (P1X 면), FWP 가 g5w 를 모든 벌점 커널에 넘김
        print(f"[절 CM95 gpw] 씬 결정: 모드 {ST['mode']} ({'A 모든 씬' if ST['mode'] == 1 else 'B κ 0 씬만'}), κ {k:g}, lightg3 {on} -> "
              f"{'P1X 면 가중 w_F = 1 − φ_min (' + ('lightg3 가중의 P1X 몫에 곱' if on else 'ghost6 방식 5') + ')' if app else '아무것도 안 함 (H5 그대로)'}", flush=True)
        _info()
        return on

    def weights(s, ok, pm):
        o_w(s, ok, pm)
        if ST["path"] == "mul":
            ok(s, "gq_gpw", int(s.capQc), (P(s.IP), P(s.cb), P(s.cidx), P(s.g3fm), P(s.g5phi), P(s.l3id), P(s.g5w)), 128)
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
                    if DG and getattr(s, "g3fm", None) is not None:
                        diag(s, fi)
                    if ST["nfr"] <= 2 or ST["nfr"] % 10 == 0:
                        _info()
                except Exception:
                    import traceback
                    with open(os.path.join(out, "gpw_err.txt"), "a") as f_:
                        f_.write(f"f{fi}\n" + traceback.format_exc()[-3000:] + "\n")
            return res
        return w
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)
    if hasattr(ART, "split_step"):
        ART.split_step = outer(ART.split_step)
    _info()
    print(f"[절 CM95] gpw 켬 -- 모드 {ST['mode']}, P1X 면 가중 w_F = 1 − φ_min (강성·고스트 질량), 진단 {DG}", flush=True)
