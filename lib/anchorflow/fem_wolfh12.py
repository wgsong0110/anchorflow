"""절 CM93 묶음 wolfh12 (11-09 위임): wolf H8 풀이 걸음이 압축을 받아들이는 이유 가르기 -- (c) 진단: 압축 가우스 점의 탄성 체적 강성이 기울기·헤시안에 들어가는가.

진단 (dg 1, 결과 불변 -- 읽기만, 출력 버퍼 따로): 걸음 끝 커밋 (ao_commit) 직전 같은 X = F_q + Σ U ⊗ g (커밋 RM 앞 시험 탄성 F) 로 가우스 점마다
  J = det X, tr = Σ log|σ| (Hencky 체적), st (ao_dpf: 0 탄성 · 1 원뿔 · 2 꼭짓점), 평균 키르히호프 응력 p = Σ τ_i / 3 (ao_elem 의 기울기 τ 그대로, 압축 −),
  체적 에너지 밀도 Ψ_v = ½ K₃ tr² (K₃ = λ + 2μ/3), 편차 에너지 밀도 Ψ_d = μ Σ e_i², 헤시안 체적 곡률 비 h⁺/h
  (주값 공간 D (ao_elem 의 Hencky 탄성 D 그대로) 의 체적 방향 v_i = σ_i/3 곡률: h = vᵀ D v, h⁺ = vᵀ D⁺ v, D⁺ = 음의 고유값을 0 으로 -- ao_elem 투영 그대로), V_q.
  프레임마다 w12.jsonl 한 줄: J 칸 (전체 · J < 1 · J < 0.9 · J < 0.5 · J ≤ 0) 별 점 수, Σ V, V 가중 평균 tr · p · h⁺/h · 꼭짓점 몫, Σ V Ψ_v · Σ V Ψ_d.
훅 (실행기 runpy 순간, 가장 바깥): 최종 CUDA 글자 (cupy RawModule 직전) 의 ao_commit 앞에 W12_SRC, RTSolver.k 의 ao_commit 앞, RTSolver.step (기록).
"""
import json
import os

import numpy as np
import torch

ST = dict(mode=0, dg=0, out=".", nsrc=0, nalloc=0, ncall=0, nframe=0, err=None)
MARK = "// ---- 절 CM93 wolfh12"
LOG = [None]


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


W12_SRC = r"""
// ---- 절 CM93 wolfh12: 걸음 끝 (커밋 앞) 가우스 점 체적 진단 (읽기만) ----
__global__ void w12_q(const int* IP, const int* cb, const float* BG, const float* Fq, const float* Vq, const float* U, float mu, float lam, float cdp, float* o) {
    int q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= IP[12]) return;
    int oc = q & 7;
    int ids[27]; qnodes(q, IP, cb, ids);
    float F[9]; for (int k=0;k<9;k++) F[k] = Fq[(size_t)q*9+k];
    float X[9]; for (int k=0;k<9;k++) X[k] = F[k];
    for (int o_=0;o_<27;o_++) { float g[3]; gqn(F, BG + (oc*27+o_)*3, g); int id = ids[o_];
        for (int a=0;a<3;a++) { float ua = U[id*3+a]; X[a*3] += ua*g[0]; X[a*3+1] += ua*g[1]; X[a*3+2] += ua*g[2]; } }
    float J = X[0]*(X[4]*X[8]-X[5]*X[7]) - X[1]*(X[3]*X[8]-X[5]*X[6]) + X[2]*(X[3]*X[7]-X[4]*X[6]);
    float Uu[9], S[3], V[9];
    ssvdf(X, Uu, S, V);
    float sgn[3], sg[3], ep[3], e[3], tr, rho, s;
    int st = ao_dpf(S, cdp, sgn, sg, ep, e, tr, rho, s);
    float K3 = lam + 2.f*mu/3.f;
    float ss[3], pe[3], ts = 0.f;
    for (int i=0;i<3;i++) {
        float tau = (st == 2) ? 0.f : ((st == 1) ? (K3*tr + 2.f*mu*s*e[i]) : (2.f*mu*ep[i] + lam*tr));
        ts += tau; ss[i] = sgn[i]*sg[i]; pe[i] = (2.f*mu*ep[i] + lam*tr)/ss[i];
    }
    float D[3][3], Dk[3][3], Ev[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) { D[i][j] = (((i==j) ? 2.f*mu : 0.f) + lam)/(ss[i]*ss[j]) - ((i==j) ? pe[i]/ss[i] : 0.f); Dk[i][j] = D[i][j]; }
    float v[3] = {ss[0]/3.f, ss[1]/3.f, ss[2]/3.f};
    float h = 0.f; for (int i=0;i<3;i++) for (int j=0;j<3;j++) h += v[i]*Dk[i][j]*v[j];
    jac3f(D, Ev);
    float hp = 0.f;
    for (int m=0;m<3;m++) { float dm = fmaxf(D[m][m], 0.f); if (!(dm > 0.f) || !isfinite(dm)) continue;
        float c = Ev[0][m]*v[0] + Ev[1][m]*v[1] + Ev[2][m]*v[2]; hp += dm*c*c; }
    float* r = o + (size_t)q*8;
    r[0] = J; r[1] = tr; r[2] = (float)st; r[3] = ts/3.f; r[4] = 0.5f*K3*tr*tr; r[5] = mu*(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    r[6] = (h != 0.f) ? hp/h : 0.f; r[7] = Vq[q];
}
"""
HEAD = "__global__ void ao_commit("


def patch(code):
    if MARK in code or HEAD not in code:
        return code
    i = code.index(HEAD)
    ST["nsrc"] += 1
    return code[:i] + MARK + "\n" + W12_SRC + code[i:]


def _dump():
    try:
        with open(os.path.join(ST["out"], "w12info.json"), "w") as f_:
            json.dump(ST, f_, ensure_ascii=False, default=str)
    except Exception:
        pass


def _grp(B, m):
    n = int(m.sum())
    if n == 0:
        return dict(n=0)
    b = B[m].double(); V = b[:, 7]; sv = float(V.sum())
    wm = (lambda c: float((b[:, c] * V).sum() / sv)) if sv > 0 else (lambda c: float(b[:, c].mean()))
    return dict(n=n, V=sv, tr=wm(1), p=wm(3), hr=wm(6), apex=float(((b[:, 2] == 2).double() * V).sum() / sv) if sv > 0 else None,
                cone=float(((b[:, 2] == 1).double() * V).sum() / sv) if sv > 0 else None,
                Ev=float((b[:, 4] * V).sum()), Ed=float((b[:, 5] * V).sum()), Jmin=float(b[:, 0].min()))


def _hook(R):
    oa, ok, os_ = R.alloc, R.k, R.step

    def alloc(s, need):
        r = oa(s, need)
        if ST["dg"] and getattr(s, "_mat", None) == 2 and getattr(s, "Fq", None) is not None:          # 모래 (MatRT mat 2) 만
            capQ = s.Fq.numel() // 9
            if getattr(s, "w12b", None) is None or s.w12b.numel() != capQ * 8:
                s.w12b = torch.zeros(capQ * 8, device=s.dev, dtype=torch.float32)
                ST["nalloc"] += 1
        return r

    def k(s, name, n, args, b=128):
        if name == "ao_commit" and getattr(s, "w12b", None) is not None:
            ok(s, "w12_q", n, tuple(args[:4]) + (P(s.Vq), args[4], np.float32(s.mu), np.float32(s.lam), args[5], P(s.w12b)), 128)
            ST["ncall"] += 1
        return ok(s, name, n, args, b)

    def step(s, *a, **kw):
        r = os_(s, *a, **kw)
        if getattr(s, "w12b", None) is not None:
            ST["nframe"] += 1
            if LOG[0] is not None:
                try:
                    nq = int(s.IP[12].item())
                    B = s.w12b[:nq * 8].view(nq, 8)
                    B = B[B[:, 7] > 0]
                    J = B[:, 0]
                    o = dict(f=ST["nframe"], nq=nq, mu=float(s.mu), lam=float(s.lam), cdp=float(s.cdp),
                             all=_grp(B, torch.ones_like(J, dtype=torch.bool)), c1=_grp(B, J < 1), c09=_grp(B, J < 0.9), c05=_grp(B, J < 0.5), inv=_grp(B, J <= 0))
                    LOG[0].write(json.dumps(o) + "\n"); LOG[0].flush()
                except Exception as e:
                    ST["err"] = repr(e)
            if ST["nframe"] in (1, 2, 5) or ST["nframe"] % 25 == 0:
                _dump()
        return r

    R.alloc, R.k, R.step = alloc, k, step


def install(out, mode=0, dg=0):
    from . import fem_an_rt as ART
    ST.update(mode=int(mode), dg=int(dg), out=out)
    if ST["dg"]:
        LOG[0] = open(os.path.join(out, "w12.jsonl"), "w")
    cpm = ART.cp
    orm = cpm.RawModule

    def RawModule(*a, **kw):
        if "code" in kw:
            kw["code"] = patch(kw["code"])
        elif a:
            a = (patch(a[0]),) + tuple(a[1:])
        return orm(*a, **kw)
    cpm.RawModule = RawModule
    _hook(ART.RTSolver)
    print(f"[절 CM93] wolfh12 켬 -- 모드 {ST['mode']}, 진단 {ST['dg']}", flush=True)
    _dump()
