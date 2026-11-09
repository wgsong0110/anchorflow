"""절 CM82 묶음 speed11 (11-08 위임, 수박 X3c 두 시도 프레임): 공통 후보 H5 (절 CM78) 위 진단 + 고침 후보. 감싸개 exe/speed11_wrap.py 가 부름.

(D) 진단 (--s11dg 1, 결과 불변 -- 읽기·따로 둔 버퍼만): X3g 경계 ('r' = 첫 시도 K 1 반복 0 을 푼 뒤 반복 1 시작, 절 CM74) 에서 호스트가 판정 값을 읽는 순간
    (speed4 replay 의 int(s._xbn[3]) -- 판정 여부와 무관하게 걸음마다) 그래프 밖에서:
    - 잔차: ‖g0‖²/F² (반복 0 시작, CD[0]/CD[2]), ‖g1‖²/F² (반복 1 시작 = X3g 판정 값, SC[1]/CD[2]), g1ᵀD⁻¹g1 / F² (SC[0] = 블록 야코비 전처리 노름)
    - PCG (반복 0 의 풀이): CT[3] 반복 수, CT[5] 상한까지 간 풀이 수 (op 7: 허용치 못 맞춤)
    - 보폭 (반복 0): SC[14] = 2^-km (노드별 장벽 보폭 최댓값 지수), 전역 뒤집힘 필터 지수 kg, 장벽 kb, 노드별 kn > kg 몫
    - g1 의 노드 몫: 질량 무리 (m < m̄ 고정 후보 · m̄ ≤ m < 50 m̄ 튐 띠 · 무거움; 보고용), 국소로 더 잘린 노드 (kn > kg), 가벼움 벌점 면 노드
    - 벌점 몫: ghost6 gq_force 를 따로 둔 버퍼에 (같은 커널·같은 k·cm·y·ŷ) 면 가중 (가) 가벼움 몫 fw − 1[P1X] (lightg3 의 l3id), (나) P1X 몫 1[P1X]
      -> ‖T f_L‖²/F², ‖T f_P‖²/F², ‖g1 − T f_L‖²/‖g1‖², 블록 대각 몫 tr B_L / (3 M/τ² + tr B) (B = 반복 1 기울기 때 실행기 블록 대각, 벌점 포함)
    -> <out>/s11.jsonl (걸음마다 한 줄, 'trig' = X3g 로 넘어갔는지).

(Z) 시작점 고착 역사상 고침 (--s11z 1|2, 결과 바뀜 -- 후보): 진단 결과 무너진 걸음은 걸음 시작부터 ‖g0‖²/F² ~ 1e24 이고 그 잔차의 100 % 가 바닥 고착 노드 (pm 1)
    의 관성 몫 Mi η -- 고착 몫 √k = s 가 0 이 아닌 아주 작은 값 (1e-19, k = (1 − r)^kex 가 f32 바닥 근처) 인 노드에서 시작점 기준 η_z = T⁻¹ (z0 − ŷ) 가
    (1/s) dd 로 1e15 m 가 됨 (aw_y0z_a 의 1/s, s = 0 일 때만 0 -- s → 0 에서 불연속). 시작점 섞기 β < 1 이면 η = η_z + β (η − η_z) 로 그대로 들어감.
    고침 = 시작점 기준의 고착 방향 성분을 1/s 대신 s 로: T η_z = k P (z0 − ŷ) (물리 변위 = 고착 몫 k 만큼 영속도 위치 쪽, k = 1 이면 원래와 같고 k → 0 이면 연속으로 ŷ).
    ao_floor_n 의 목표 h = Ũ + (1 − k) P d (고착 몫 1 − k 만큼 묶음) 와 같은 k 섞기. 문턱 없음.
    1 = aw_y0z_a (옛 실행기 시작점 섞기 기준) 만 · 2 = 1 + 걸음 시작 위치 시작점 (옛 ao_floor_n --y0u · 새 rt_ga_y0 의 (s > 1e-6) ? 1/s : 0 도 s 로; 모든 씬).
    (s1k: 1·2 는 수박 첫 접촉 f3 부터 바닥을 뚫음 (부분 고착 노드의 법선 시작점이 z0 에 못 닿아 장벽 밖) -> 버림.)
    3 = aw_y0z_a 의 고착 판정을 걸음 시작 위치 시작점 두 커널 (ao_floor_n --y0u, rt_ga_y0) 이 이미 쓰는 정의 (√k > 1e-6 이면 1/√k, 아니면 그 성분 0 = ŷ, 절 AN.err3 처치 (a)) 와 같게 -- 새 값 아님.
"""
import json
import math
import os

import numpy as np
import torch


class C:
    out = "."
    dg = 0
    z = 0
    log = None
    said = False
    nerr = 0
    rec = []
    fs = None


def P(t):
    return np.uint64(0) if t is None else np.uint64(t.data_ptr())


def _skip_get(s):
    import cupy as cp
    vals = []
    for mod in (s._mods() if hasattr(s, "_mods") else [s.mod]):
        try:
            g = mod.get_global("AF_SKIP")
            a = cp.ndarray((1,), dtype=cp.int32, memptr=g)
            vals.append((a, int(a[0])))
        except Exception:
            pass
    return vals


def _q(v, qs=(0.5, 0.9, 1.0)):
    if v.numel() == 0:
        return None
    v = v.double().sort().values
    return [float(v[min(int(q * (v.numel() - 1) + 0.5), v.numel() - 1)]) for q in qs]


def _pen(s, fw, G6):
    """따로 둔 버퍼에 벌점 힘 (f64) · 블록 대각 (f32): 실행기 ELEM 훅과 같은 인자 (k = s._gpk, cm = 걸음 안 k_M/τ²)."""
    capN, capQc = int(s.capN), int(s.capQc)
    f_ = torch.zeros(capN * 3, device=s.dev, dtype=torch.float64)
    B_ = torch.zeros(capN * 9, device=s.dev, dtype=torch.float32)
    k = float(s._gpk)
    cm = float(G6.cm_of(s))
    yh = s.yh if cm != 0.0 else None
    s.k("gq_force", capQc, (P(s.IP), P(s.cb), P(s.g3fm), P(fw), np.float32(k), np.float32(cm), P(s.y), P(yh), P(None), P(f_), P(B_), P(None)), 128)
    return f_, B_


def rdiag(s, trig):
    from . import fem_ghost6 as G6
    from . import fem_ghost2 as G2
    from . import fem_speed6 as S6
    torch.cuda.synchronize()
    sk_vals = _skip_get(s)
    for a, _ in sk_vals:
        a[0] = 0
    s._g3chk = True
    s._l3in = True
    try:
        IP = [int(q) for q in s.IP.cpu().tolist()]
        nn = IP[9]
        SC = s.SC.double().cpu().numpy()
        CD = s.CD.double().cpu().numpy()
        CT = s.CT.cpu().numpy()
        F2 = float(CD[2])
        r = dict(f=int(S6.W.fr), trig=int(trig), g0F=float(CD[0]) / F2 if F2 > 0 else None, g1F=float(SC[1]) / F2 if F2 > 0 else None,
                 rz1F=float(SC[0]) / F2 if F2 > 0 else None, F=math.sqrt(F2) if F2 > 0 else None,
                 pcg=int(CT[3]), pcap=int(CT[5]), nsol=int(CT[7]), st=float(SC[14]), nn=nn)
        mn = s.mn.reshape(-1)[:nn].double()
        act = (mn > 0).double()[:, None]                                                     # 압축 판 (rt_grad_c) 은 질량 있는 노드만 씀 -- 빈 노드 값은 지난 것
        g = -s.g0buf.reshape(-1)[:nn * 3].double().reshape(nn, 3) * act
        g2 = (g * g).sum(1)
        G = float(g2.sum())
        mb = float(getattr(s, "mthr", 0.0) or 0.0)
        cls = torch.where(mn < mb, 0, torch.where(mn < 50 * mb, 1, 2))
        r["gsh_m"] = [float(g2[cls == c].sum()) / G if G > 0 else None for c in range(3)]
        r["n_m"] = [int((cls == c).sum()) for c in range(3)]
        kn = getattr(s, "wskn", None)
        if kn is not None:
            kn = kn.reshape(-1)[:nn].long()
            kg = int(s.wskg[0]); kb = int(s.kk[0]) if getattr(s, "kk", None) is not None else -1
            cut = kn > kg
            r.update(kg=kg, kb=kb, kmax=int(kn.max()) if nn else 0, ncut=int(cut.sum()), gsh_cut=float(g2[cut].sum()) / G if G > 0 else None)
        i = int(torch.argmax(g2)) if nn else 0
        it2_ = float(getattr(s, "_g3it2", 0.0))
        eta = s.eta.reshape(-1)[:nn * 3].double().reshape(nn, 3) * act
        gi = mn[:, None] * it2_ * eta                                                       # 관성 몫 Mi η (g = Mi η + T f)
        r["gsh_in"] = float(((gi) ** 2).sum()) / G if G > 0 else None
        r["gsh_f"] = float(((g - gi) ** 2).sum()) / G if G > 0 else None
        skv = s.sk.reshape(-1)[:nn].double(); pmv = s.pm.reshape(-1)[:nn]
        top10 = torch.topk(g2, min(10, nn)).indices if nn else None
        if top10 is not None:
            dy = (s.y.reshape(-1)[:nn * 3] - s.yh.reshape(-1)[:nn * 3]).double().reshape(nn, 3)
            r["top10"] = dict(g2sh=float(g2[top10].sum()) / G if G > 0 else None, sk=[float(q) for q in skv[top10]], pm=[int(q) for q in pmv[top10]],
                              eta=[float(q) for q in eta[top10].norm(dim=1)], dy=[float(q) for q in dy[top10].norm(dim=1)], m_mb=[float(q) / mb if mb > 0 else None for q in mn[top10]],
                              ginsh=float((gi[top10] ** 2).sum()) / max(float(g2[top10].sum()), 1e-300))
            r["n_pm"] = [int((pmv == c).sum()) for c in range(4)]
            r["gsh_pm"] = [float(g2[pmv == c].sum()) / G if G > 0 else None for c in range(4)]
        r["top"] = dict(g2sh=float(g2[i]) / G if G > 0 else None, m_mb=float(mn[i]) / mb if mb > 0 else None, pm=int(s.pm.reshape(-1)[i]), kn=int(kn[i]) if kn is not None else None)
        fw = getattr(s, "g5w", None)
        idv = getattr(s, "l3id", None)
        if fw is not None and idv is not None and getattr(s, "_gpk", None) is not None:
            nrm = torch.tensor(s.floor["NR"] if getattr(s, "floor", None) else [0.0, 0.0, 1.0], device=s.dev, dtype=torch.float64)
            sk = s.sk.reshape(-1)[:nn].double(); pm = s.pm.reshape(-1)[:nn]
            fw = fw.reshape(-1); idv = idv.reshape(-1)[:fw.numel()]
            fwL = (fw - idv).contiguous()
            fwP = idv.contiguous()
            out = {}
            for nm_, w_ in (("L", fwL), ("P", fwP)):
                f_, B_ = _pen(s, w_, G6)
                torch.cuda.synchronize()
                Tf = G2._T(f_[:nn * 3].reshape(nn, 3), sk, pm, nrm) * act
                out[nm_] = (Tf, B_[:nn * 9].double().reshape(nn, 9) * act)
            TfL, BL = out["L"]; TfP, BP = out["P"]
            r["fL2F"] = float((TfL * TfL).sum()) / F2 if F2 > 0 else None
            r["fP2F"] = float((TfP * TfP).sum()) / F2 if F2 > 0 else None
            r["g_noL"] = float(((g - TfL) ** 2).sum()) / G if G > 0 else None
            r["g_noP"] = float(((g - TfP) ** 2).sum()) / G if G > 0 else None
            it2 = float(getattr(s, "_g3it2", 0.0))
            Bt = s.B.reshape(-1)[:nn * 9].double().reshape(nn, 9) * act
            trT = 3.0 * mn * it2 + Bt[:, 0] + Bt[:, 4] + Bt[:, 8]
            trL = BL[:, 0] + BL[:, 4] + BL[:, 8]
            trP = BP[:, 0] + BP[:, 4] + BP[:, 8]
            on = trL > 0
            r["nL"] = int(on.sum())
            r["gsh_L"] = float(g2[on].sum()) / G if G > 0 else None
            shL = (trL / trT.clamp_min(1e-300))[on]
            r["hL_q"] = _q(shL)
            r["hL_gw"] = float((shL * g2[on]).sum() / g2[on].sum()) if bool(on.any()) and float(g2[on].sum()) > 0 else None
            r["hLM_q"] = _q((trL / (3.0 * mn * it2).clamp_min(1e-300))[on])
            r["hP_q"] = _q((trP / trT.clamp_min(1e-300))[trP > 0])
            if kn is not None:
                r["ncutL"] = int((cut & on).sum())
        C.rec.append(r)
        C.log.write(json.dumps(r, ensure_ascii=False) + "\n")
        C.log.flush()
    except Exception:
        import traceback
        C.nerr += 1
        if C.nerr <= 5:
            with open(os.path.join(C.out, "s11_err.txt"), "a") as f_:
                f_.write(traceback.format_exc()[-3000:] + "\n")
    finally:
        s._g3chk = False
        s._l3in = False
        torch.cuda.synchronize()
        for a, v in sk_vals:
            a[0] = v
        torch.cuda.synchronize()


ZP = [("aw_y0z_a", "    float isk = (s > 0.f) ? 1.f/s : 0.f;\n    float dd[3]; for (int a=0;a<3;a++) dd[a] = z0[n*3+a] - yh[n*3+a];",
       "    float isk = s;   // 절 CM82 Z: T η_z = k P dd (1/s 대신)\n    float dd[3]; for (int a=0;a<3;a++) dd[a] = z0[n*3+a] - yh[n*3+a];", 1),
      ("ao_floor_n y0uk", "        float isk = (s > 1e-6f) ? 1.f/s : 0.f;\n        float dd[3]; for (int a=0;a<3;a++) dd[a] = U[n*3+a] - h[a];",
       "        float isk = s;   // 절 CM82 Z2\n        float dd[3]; for (int a=0;a<3;a++) dd[a] = U[n*3+a] - h[a];", 2),
      ("rt_ga_y0", "    float isk = (s > 1e-6f) ? 1.f/s : 0.f;\n    float dd[3]; for (int a=0;a<3;a++) dd[a] = U[n*3+a] - yh[n*3+a];",
       "    float isk = s;   // 절 CM82 Z2\n    float dd[3]; for (int a=0;a<3;a++) dd[a] = U[n*3+a] - yh[n*3+a];", 2)]
ZP.append(("aw_y0z_a 3", "    float isk = (s > 0.f) ? 1.f/s : 0.f;\n    float dd[3]; for (int a=0;a<3;a++) dd[a] = z0[n*3+a] - yh[n*3+a];",
           "    float isk = (s > 1e-6f) ? 1.f/s : 0.f;   // 절 CM82 Z3: ao_floor_n --y0u · rt_ga_y0 와 같은 고착 판정 (√k ≤ 1e-6 이면 그 성분 0 = ŷ)\n    float dd[3]; for (int a=0;a<3;a++) dd[a] = z0[n*3+a] - yh[n*3+a];", 3))
ZST = dict(done=set(), said=set())


def _zpatch(src):
    for nm, a, b, lv in ZP:
        if ((C.z == 3 and lv == 3) or (C.z < 3 and C.z >= lv)) and a in src:
            src = src.replace(a, b)
            ZST["done"].add(nm)
    return src


def _install_z():
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    ob, oa = ART.build_src, AORT._src

    def bs(*a, **kw):
        r = ob(*a, **kw)
        r2 = _zpatch(r) if isinstance(r, str) else r
        print(f"[절 CM82] Z 고침 (새 실행기) {sorted(ZST['done'])}", flush=True)
        return r2

    def az(*a, **kw):
        r = oa(*a, **kw)
        r2 = _zpatch(r) if isinstance(r, str) else r
        print(f"[절 CM82] Z 고침 {sorted(ZST['done'])}", flush=True)
        return r2
    ART.build_src, AORT._src = bs, az


class _XBN:
    """s._xbn (고정 호스트 int32 4 칸의 numpy 보기) 를 감쌈: replay 의 'r' 경계 판정 읽기 (int(s._xbn[3])) 순간 진단."""
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
    o = S4._xb_buf

    def _xb_buf(s):
        o(s)
        if not isinstance(s._xbn, _XBN):
            s._xbn = _XBN(s, s._xbn)
    S4._xb_buf = _xb_buf


def install(out, dg=0, z=0):
    """실행기를 runpy 로 띄우는 순간 (speed6~10 설치 뒤 = 가장 바깥) 에 부름."""
    C.out, C.dg, C.z = out, int(dg), int(z)
    if C.z:
        _install_z()
    if C.dg:
        C.log = open(os.path.join(out, "s11.jsonl"), "w")
        _install_dg()
    print(f"[절 CM82] speed11 켬 진단 {C.dg} 시작점 고착 고침 Z {C.z}", flush=True)
