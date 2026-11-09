"""절 CM85 묶음 speed12 (11-08 위임, 수박 X3c 두 시도 프레임): 공통 후보 H7 (절 CM84) 위 시작점 기준 고침 + 진단. 감싸개 exe/speed12_wrap.py 가 부름. 옛 실행기 (수박, AORTSolver) 만 바뀜.

배경 (절 CM82): 두 시도 프레임 = 걸음 시작점 섞기 기준 (aw_y0z_a) 이 바닥 고착 노드에서 η_z = T⁻¹ P (U − ŷ) = (1/√k) P (U − ŷ) 로 나눠
    √k ≈ 1e-19 인 노드에서 η 가 1e11~1e16 이 되고, 섞기 β < 1 이면 η = η_z + β (η − η_z) 로 들어가 관성 잔차 Mi η 가 ‖g‖²/F² ~ 1e24 -> X3g 가 X2 (고착 상한 + 속도 수준 접촉) 로 다시 함.
    이 무너짐을 없앤 Z1~Z3 은 첫 접촉에서 X2 가 안 켜져 바닥을 뚫음 (이 무너짐이 X2 를 켜는 신호).

(Z) --s12z 1 (결과 바뀜): 시작점 기준과 X2 신호를 가름.
    (가) 기준 η_z 를 1/√k 로 나누지 않음: 고착 방향 성분 η_z = √k P (U − ŷ) (T η_z = k P (U − ŷ); k = 1 이면 원래와 같고 k -> 0 이면 연속으로 ŷ; 절 CM82 Z1 과 같은 식). 문턱 없음.
    (나) X2 신호는 따로: 옛 기준이 시작점에 넣던 관성 잔차 S = (1 − β)² Σ_{√k > 0} (M/τ²)² |T⁻¹ P (U − ŷ)|² (옛 aw_y0z_a 의 1/√k 와 같은 노드 집합 (√k > 0),
        β = aw_y0z_b 의 섞기 보폭 (β = 1 이면 기준 몫 0)) 를 노드 커널 안에서 f64 로 모아, 반복 0 시작 판정 값 ‖g0‖² 에 더해 같은 문턱 (X3g 의 T = 1e6, 새 값 아님) 과 비교:
        !(‖g0‖² + S ≤ T F²) 이면 X2 를 켜고 걸음 시작부터 (절 CM50 X3e4 / CM82 E 의 반복 0 'r' 경계·되돌림 그대로). X3g 의 반복 1 경계는 끔 (경계는 하나).
        = 옛 판정 (반복 0 시작 ‖g0‖²/F² > 1e6 과 X3g 판정이 236/236 같음, 절 CM82.1) 을 시작점에 큰 수를 넣지 않고 그대로 계산.
(D) --s12dg 1 (결과 불변 -- 읽기만): 'r' 경계 (z 1 은 반복 0, z 0 은 X3g 반복 1) 에서 걸음마다 S/F², ‖g0‖²/F², β, 판정 + 바닥 노드 고착 몫 분포
    (√k = 0 · 0 < √k ≤ 1e-6 · 1e-6 < √k < 1 노드 수, 띠 질량 몫 r = an/mqn (= 1 이면 k 0), 노드 질량 / m̄, 거듭제곱 KEX) -> <out>/s12.jsonl.
    z 0 에서도 S 는 같은 커널로 모음 (판정에는 안 씀) -> H7 의 X3g 판정과 '‖g0‖² + S' 판정이 같은지 대조.
"""
import json
import os

import numpy as np
import torch


class C:
    out = "."
    z = 0
    dg = 0
    log = None
    mod = None
    fn = {}
    done = set()
    nrec = 0
    ntrig = 0
    nerr = 0
    kex = None


A_OLD = "    float isk = (s > 0.f) ? 1.f/s : 0.f;\n    float dd[3]; for (int a=0;a<3;a++) dd[a] = z0[n*3+a] - yh[n*3+a];"
SIG = (
    "\n    if (s > 0.f) { float dn_ = dd[0]*n0 + dd[1]*n1 + dd[2]*n2; double q_ = 0.0;   // 절 CM85 (나): 옛 기준의 관성 몫 (M/τ²)² |T⁻¹ P dd|² 의 M² |P dd|² / k (τ 는 판정 커널에서)\n"
    "        if (p == 0) q_ = (double)dd[0]*dd[0] + (double)dd[1]*dd[1] + (double)dd[2]*dd[2];\n"
    "        else if (p == 3) q_ = (double)dn_*dn_;\n"
    "        else { double t0 = (double)dd[0] - (double)dn_*n0, t1 = (double)dd[1] - (double)dn_*n1, t2 = (double)dd[2] - (double)dn_*n2; q_ = t0*t0 + t1*t1 + t2*t2; }\n"
    "        double m_ = (double)mn[n], s_ = (double)s; atomicAdd(&S12A, m_*m_*q_/(s_*s_)); }")
A_NEW1 = "    float isk = s;   // 절 CM85 (가): 기준 η_z = √k P dd (T η_z = k P dd) -- 1/√k 로 나누지 않음\n    float dd[3]; for (int a=0;a<3;a++) dd[a] = z0[n*3+a] - yh[n*3+a];" + SIG
A_NEW0 = A_OLD + SIG
F_OLD = ("        k = expf(kex*log1pf(-r));\n        float vv = al*((y[n*3]-z0[n*3])*n0 + (y[n*3+1]-z0[n*3+1])*n1 + (y[n*3+2]-z0[n*3+2])*n2);\n"
         "        p = (vout && vv > 0.f) ? 1 : 0;\n        if (slip)")
F_NEW = ("        k = expf(kex*log1pf(-r));\n        float vv = al*((y[n*3]-z0[n*3])*n0 + (y[n*3+1]-z0[n*3+1])*n1 + (y[n*3+2]-z0[n*3+2])*n2);\n"
         "        p = (vout && vv > 0.f) ? 1 : 0;\n        if (n == 0) S12A = 0.0;   // 절 CM85: 걸음마다 S 비움 (aw_y0z_a 는 뒤 커널)\n        if (slip)")


def _patch(src):
    if not isinstance(src, str) or "aw_y0z_a" not in src:
        return src
    assert src.count(A_OLD) == 1 and src.count(F_OLD) == 1, ("절 CM85 패치 자리", src.count(A_OLD), src.count(F_OLD))
    src = src.replace('extern "C" {', 'extern "C" {\n__device__ double S12A;', 1)
    src = src.replace(A_OLD, A_NEW1 if C.z else A_NEW0).replace(F_OLD, F_NEW)
    C.done.add("aw_y0z_a z" if C.z else "aw_y0z_a S 만")
    return src


R_SRC = r"""
extern "C" {
__global__ void s12_r(const int* IP, const double* SC, const double* CD, const double* SA, const double* bt, double it2, double T, int dec, int* out, double* v) {
    double b = bt[0], w = (1.0 - b)*(1.0 - b);                  // aw_y0z_b: β = 1 이면 η 그대로 (기준 몫 0)
    double S = w*it2*it2*SA[0];
    v[0] = SC[1]; v[1] = CD[2]; v[2] = S; v[3] = b; v[4] = SA[0]; v[5] = it2;
    if (dec) { out[2] = IP[13]; out[3] = !(SC[1] + S <= T*CD[2]) ? 1 : 0; }
}
}
"""


def _P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _load():
    import cupy as cp
    if C.mod is None:                                                                  # 캡처 밖에서 적재 (캡처 중 모듈 적재는 CUDA 가 거절)
        C.mod = cp.RawModule(code=R_SRC)
        C.fn["r"] = C.mod.get_function("s12_r")


def _rk(args):
    from . import fem_rt_cuda as RTC
    _load()
    with RTC._Stream():
        C.fn["r"]((1,), (1,), args)


def _sa_ptr(s):
    c = getattr(s, "_s12sa", None)
    if c is not None and c[0] is s.mod:
        return c[1]
    p = _sa_ptr0(s)
    s._s12sa = (s.mod, p)
    return p


def _sa_ptr0(s):
    mods = [s.mod] + (list(s._mods()) if hasattr(s, "_mods") else [])
    for m in mods:
        try:
            return np.uint64(int(m.get_global("S12A").ptr))
        except Exception:
            continue
    raise RuntimeError("S12A 없음")


def _install_src():
    from . import fem_ao_rt as AORT
    oa = AORT._src

    def az(*a, **kw):
        r = _patch(oa(*a, **kw))
        print(f"[절 CM85] 시작점 기준 패치 {sorted(C.done)}", flush=True)
        return r
    AORT._src = az


def _install_ctl():
    from . import fem_ao_rt as AORT
    from . import fem_common3 as C3
    from . import fem_speed4 as S4
    assert S4.X1.on and S4.X1.level == 2 and S4.X1.T > 0, "--s12 는 X3g (@s8g) 위"
    cls = AORT.AORTSolver
    o_ctl = cls.ctl

    def ctl(s, op, it=0):
        cap = getattr(s, "_c3cap", None) is not None
        k1 = getattr(s, "_x1K", 0) == 1
        if not cap:                                                                    # 캡처 밖 (데움): 모듈 적재 · S12A 주소
            _load()
            if getattr(s, "mod", None) is not None:
                try:
                    _sa_ptr(s)
                except Exception:
                    pass
        if C.z and op == 3 and it == 1 and cap and k1:                                   # X3g 반복 1 경계 끔 (경계는 반복 0 하나)
            s._x1K = -1
            try:
                return o_ctl(s, op, it)
            finally:
                s._x1K = 1
        r = o_ctl(s, op, it)
        if op == 3 and it == 0 and cap and k1 and getattr(s, "_xbd", None) is not None and getattr(s, "awb", None) is not None:
            if getattr(s, "_s12v", None) is None:
                s._s12v = torch.zeros(8, device=s.dev, dtype=torch.float64)
            _rk((_P(s.IP), _P(s.SC), _P(s.CD), _sa_ptr(s), _P(s.awb), np.float64(1.0 / (s.tau * s.tau)), np.float64(S4.X1.T), np.int32(1 if C.z else 0), _P(s._xbd), _P(s._s12v)))
            if C.z:
                S4._d2h(s)
                C3.mark(s, "r")
        return r
    cls.ctl = ctl


def rdiag(s, trig):
    try:
        from . import fem_speed6 as S6
        fr = int(getattr(S6.W, "fr", -1))
    except Exception:
        fr = -1
    try:
        v = s._s12v.cpu().tolist() if getattr(s, "_s12v", None) is not None else [float("nan")] * 8
        n = int(s.IP[9].item()) if torch.is_tensor(s.IP) else int(s.IP[9])
        sk, pm, an, mqn, mn = s.sk[:n], s.pm[:n], s.an[:n], s.mqn[:n], s.mn[:n]
        fr_ = pm != 2
        fl = (an > 0) & fr_
        r = torch.where(mqn > 0, torch.clamp(an / torch.where(mqn > 0, mqn, torch.ones_like(mqn)), max=1.0), torch.zeros_like(an))
        win = fl & (sk > 0) & (sk <= 1e-6)
        zero = fl & (sk == 0)
        mid = fl & (sk > 1e-6) & (sk < 1)
        mb = float(mn[mn > 0].mean())

        def q(t):
            if t.numel() == 0:
                return None
            t = t.double()
            return [round(float(torch.quantile(t, p)), 6) for p in (0.0, 0.5, 1.0)]
        F2 = v[1] if v[1] > 0 else float("nan")
        rec = dict(fr=fr, trig=int(trig), g0F=v[0] / F2, SF=v[2] / F2, beta=v[3], nfl=int(fl.sum()), n0=int(zero.sum()), nwin=int(win.sum()), nmid=int(mid.sum()),
                   nsat=int((fl & (r >= 1)).sum()), r_win=q(r[win]), r_mid=q(r[mid]), r_fl=q(r[fl]), m_win=q(mn[win] / mb), m_zero=q(mn[zero] / mb),
                   nlight_win=int((win & (mn < mb)).sum()), nlight_fl=int((fl & (mn < mb)).sum()), kex=C.kex)
        C.log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        C.log.flush()
        C.nrec += 1
        C.ntrig += int(trig)
    except Exception:
        C.nerr += 1
        if C.nerr <= 3:
            import traceback
            with open(os.path.join(C.out, "s12_err.txt"), "a") as f_:
                f_.write(traceback.format_exc()[-3000:] + "\n")


class _XBN:
    """s._xbn (고정 호스트 int32 4 칸) 감쌈: replay 의 'r' 경계 판정 읽기 (int(s._xbn[3])) 순간 진단 (speed11 과 같은 꼴)."""
    def __init__(self, s, a):
        self.s, self.a = s, a

    def __getitem__(self, i):
        v = self.a[i]
        if i == 3 and int(self.a[2]) == 0:
            rdiag(self.s, int(v))
        return v

    def __setitem__(self, i, v):
        self.a[i] = v

    def __len__(self):
        return len(self.a)


def _install_dg():
    from . import fem_speed4 as S4
    from . import fem_ao_rt as AORT
    o = S4._xb_buf

    def _xb_buf(s):
        o(s)
        if not isinstance(s._xbn, _XBN):
            s._xbn = _XBN(s, s._xbn)
            fl = getattr(s, "floor", None)
            C.kex = float(fl["KEX"]) if isinstance(fl, dict) and "KEX" in fl else None
    S4._xb_buf = _xb_buf


def install(out, z=0, dg=0):
    """실행기를 runpy 로 띄우는 순간 (안쪽 감싸개들 설치 뒤 = 가장 바깥) 에 부름."""
    C.out, C.z, C.dg = out, int(z), int(dg)
    _install_src()
    _install_ctl()
    if C.dg:
        C.log = open(os.path.join(out, "s12.jsonl"), "w")
        _install_dg()
    with open(os.path.join(out, "s12info.json"), "w") as f_:
        json.dump(dict(z=C.z, dg=C.dg), f_)
    print(f"[절 CM85] speed12 켬 시작점 기준 고침 z {C.z} 진단 {C.dg}", flush=True)
