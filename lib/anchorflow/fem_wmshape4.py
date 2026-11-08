"""절 CM46 묶음 wmshape4 (수박 옛 실행기 lib/anchorflow/fem_ao_rt.py 만; 감싸개 exe/wmshape4_wrap.py 인자 없으면 이 모듈을 안 부름 = 소스·그래프 그대로).

D4 진단 (--w4dg 프레임들, 결과 불변, 읽기만): 첫 접촉 프레임에서 바닥 쪽 층이 왜 안 움직이는지.
  지정 프레임 (1 부터 센 프레임 번호) 의 X3c 시도마다 (WRX.split_col 안: 끈 채 첫 시도 'try0' = WRX.on(1) 직전, 마지막 'fin' = 돌아온 뒤)
  프레임 시작 입자 높이 h (바닥 z 0.02 위) 5 cm 층마다, 입자 27 노드 B-스플라인 가중 보간으로:
    dz 관성 예측 Ut, 관성 목표 yh (ao_floor_n 의 붙은 목표 h -- X2 가 들어가는 법선 이동을 지운 뒤), 뉴턴 시작점 y0 (--y0loc 섞기 뒤),
    뉴턴 반복 it 마다 y (반복 시작), 끝 U; 띠 몫 r = an/mqn, (T 판) 도달 몫 rT, √k (sk), 시작점 β = 2^-ykn, 노드 국소 보폭 2^-wskn (반복마다),
    '전진 없음' 몫 (|dz 끝 − dz 시작점| < 0.1 mm 인 입자).  반복마다 ‖g‖²/F² (SC[1]/SC[56]).
  버퍼는 alloc 에서 잡고 복사는 커널 열에 (그래프 안) -- 풀이 값은 안 바꿈. -> <out>/w4.jsonl

T 후보 (--w4t 1, 정의 변경, X3c 의 속도 수준 접촉 X2 만 바꿈 -- 무너진 프레임 (AF_WRXON = 1) 에서만 쓰임):
  원래 X2: 띠 몫 r_n = Σ_{q: d_end < d̂} BW m_q / Σ_q BW m_q 만큼 노드의 들어가는 법선 이동을 지움 -- 띠는 걸음 끝 관성 예측 높이 d_end < d̂ 인 표본점 전부라
    떨어지는 속도 v 로 한 걸음 v·dt (첫 접촉 7.35 m/s x 0.033 s ≈ 24 cm) 안의 몸 아래쪽 전체가 r ≈ 1 -> 그 노드들의 관성 목표가 걸음 시작 자리 (멈춤).
  T: 지우는 몫을 '접촉 뒤에 갈 몫' 으로 -- 표본점 q 의 걸음 시작 높이 d_0 (x̄_q + Σ w U) 와 끝 예측 d_end 로
      α_q = (d̂ − d_end) / (d_0 − d_end)   (d_0 ≥ d̂ 일 때; 이미 d̂ 안에서 시작한 점은 α_q = 1)
    rT_n = Σ_{q 띠} BW m_q α_q / Σ_q BW m_q, X2 는 r 대신 rT 로 지움 -> 각 표본점의 관성 목표가 '바닥 (d̂) 에 닿는 자리' 까지 가고 거기서 멈춤
    (MPM 격자 바닥 조건처럼 들어가는 속도는 닿은 뒤 몫만 없앰). d_0 > d̂ > d_end 이면 0 < α < 1 이 식에서 나옴 -- 문턱·클램프·노드 바닥 고정 없음.
    띠 판정 (wb_floor_q 의 d_end < d̂)·고착 √k (X1)·장벽은 그대로. W̄ (fem_escape3 --e3w) 가 켜져 있으면 위치 판정 무게 w 는 W̄ 그대로 따라감.
"""
import json
import os

import numpy as np
import torch

ST = dict(t=False, dg=set(), out=".", log=None, cur=None, it=0, fi=None)
NIT = 48
FL_Z = 0.02
EDG = [i * 0.05 for i in range(17)]                                                    # 0 ~ 80 cm, 5 cm 층

FQ_SIG = "float pn, float dh, float* an) {"
FQ_X = "    float xq[3] = {AF_BX[q*4]/bm, AF_BX[q*4+1]/bm, AF_BX[q*4+2]/bm};\n"
FQ_L = "for (int a=0;a<3;a++) { float ye = U[id*3+a] + thinv*(y[id*3+a] - U[id*3+a]); xq[a] += w*ye; } }"
FQ_D = "    if (!(d < dh)) return;\n"
FQ_A = "    if (mw > 0.f) for (int o=0;o<27;o++) atomicAdd(&an[ids[o]], BW[oc*27+o]*mw);\n"
X2_R = "        float r_ = (mqn[n] > 0.f) ? fminf(an[n]/mqn[n], 1.f) : 0.f;\n"


def t_apply(src):
    """wb_floor_q 에 도달 몫 누적 (마지막 인자 w4an) 을 덧붙임. 장벽 판 소스가 아니면 그대로."""
    key = "__global__ void wb_floor_q("
    if key not in src:
        return src
    i = src.index(key)
    j = src.index("\n}\n", i) + 3
    b = src[i:j]
    for o_ in (FQ_SIG, FQ_X, FQ_L, FQ_D, FQ_A):
        assert b.count(o_) == 1, ("--w4t wb_floor_q", o_)
    b = b.replace(FQ_SIG, FQ_SIG[:-3] + ", float* w4an) {", 1)
    b = b.replace(FQ_X, FQ_X + "    float x0q[3] = {xq[0], xq[1], xq[2]};                                         // 절 CM46 T: 걸음 시작 위치\n", 1)
    b = b.replace(FQ_L, "for (int a=0;a<3;a++) { float ye = U[id*3+a] + thinv*(y[id*3+a] - U[id*3+a]); xq[a] += w*ye; x0q[a] += w*U[id*3+a]; } }", 1)
    b = b.replace(FQ_D, FQ_D + "    float d0_ = x0q[0]*n0 + x0q[1]*n1 + x0q[2]*n2 - pn;\n"
                  "    float al_ = (d0_ >= dh) ? (dh - d)/(d0_ - d) : 1.f;                         // 절 CM46 T: 이 걸음 이동 중 d̂ 에 닿은 뒤 몫\n", 1)
    b = b.replace(FQ_A, FQ_A + "    if (mw > 0.f && w4an) for (int o=0;o<27;o++) atomicAdd(&w4an[ids[o]], BW[oc*27+o]*(mw*al_));   // 절 CM46 T\n", 1)
    return src[:i] + b + src[j:]


def install(t=False, dg=(), out="."):
    from . import fem_ao_rt as AORT
    from . import fem_an_rt as ART
    from . import fem_wmrx as WRX
    P = ART.P
    ST.update(t=bool(t), dg=set(int(q) for q in dg), out=out)
    if ST["t"]:
        assert WRX.A2N.count(X2_R) == 1, "--w4t: X2 글자가 바뀜"
        WRX.HEAD = WRX.HEAD + "__device__ float* W4AN = 0;\n"
        WRX.A2N = WRX.A2N.replace(X2_R, "        float r_ = (mqn[n] > 0.f) ? fminf((W4AN ? W4AN[n] : an[n])/mqn[n], 1.f) : 0.f;   // 절 CM46 T: 도달 몫\n", 1)
        o_src = AORT._src

        def _src(*a, **kw):
            return t_apply(o_src(*a, **kw))
        AORT._src = _src
        print("[절 CM46 후보 T] X3c 속도 수준 접촉 X2 를 띠 몫 대신 도달 몫 (걸음 중 d̂ 에 닿은 뒤 몫) 으로", flush=True)
    o_alloc = AORT.AORTSolver.alloc

    def alloc(s, need):
        r = o_alloc(s, need)
        if ST["t"]:
            s.w4an = torch.zeros(s.capN, device=s.dev, dtype=torch.float32)
            s._w4b = None
        if ST["dg"]:
            s.w4y0 = torch.zeros(s.capN, 3, device=s.dev, dtype=torch.float32)
            s.w4yit = torch.zeros(NIT, s.capN, 3, device=s.dev, dtype=torch.float32)
            s.w4kn = torch.zeros(NIT, s.capN, device=s.dev, dtype=torch.int32)
            s.w4sc = torch.zeros(NIT, 2, device=s.dev, dtype=torch.float64)
        return r
    AORT.AORTSolver.alloc = alloc
    o_k = AORT.AORTSolver.k

    def k(s, name, n, args, b=128):
        if name == "rt_reset":
            ST["it"] = 0
            if ST["t"] and getattr(s, "mod", None) is not None and getattr(s, "w4an", None) is not None:
                key = (id(s.mod), int(s.w4an.data_ptr()))
                if s._w4b != key:
                    assert not torch.cuda.is_current_stream_capturing(), "--w4t: W4AN 은 데움 record 에서 묶어야 함"
                    try:
                        g = s.mod.get_global("W4AN")
                        ART.cp.ndarray((1,), dtype=ART.cp.uint64, memptr=g)[0] = int(s.w4an.data_ptr())
                    except Exception:
                        pass                                                           # X2 없는 모듈 (wrx 끔)
                    s._w4b = key
        if ST["t"] and name == "wb_floor_q":
            s.w4an.zero_()
            args = tuple(args) + (P(s.w4an),)
        if ST["dg"] and name in ("rt_grad", "rt_grad_c") and ST["it"] < NIT:
            s.w4yit[ST["it"]].copy_(s.y.reshape(-1, 3)[:s.capN])
        r = o_k(s, name, n, args, b)
        if ST["dg"]:
            if name == "rt_reset":                                                     # 그래프 안: 이 그래프의 반복 기록 지움 (반복 수 = F² > 0 인 줄)
                s.w4sc.zero_()
            elif name in ("yl_mix", "rt_y0mix"):
                s.w4y0.copy_(s.y.reshape(-1, 3)[:s.capN])
            elif name in ("rt_grad", "rt_grad_c") and ST["it"] < NIT:
                s.w4sc[ST["it"], 0:1].copy_(s.SC[1:2]); s.w4sc[ST["it"], 1:2].copy_(s.SC[56:57])
            elif name == "ws_sc" and ST["it"] < NIT and getattr(s, "wskn", None) is not None:
                s.w4kn[ST["it"]].copy_(s.wskn.reshape(-1)[:s.capN])
                ST["it"] += 1
        return r
    AORT.AORTSolver.k = k
    if not ST["dg"]:
        return
    ST["log"] = open(os.path.join(out, "w4.jsonl"), "w")
    o_on = WRX.on
    o_col = WRX.split_col

    def on(s, v):
        if int(v) == 1 and ST["cur"] is not None:
            try:
                snap(s, "try0")
            except Exception:
                _err()
        return o_on(s, v)
    WRX.on = on

    def split_col(s, t, h, sub_dt, depth, fn=None):
        fi = int(round(t / h)) + 1
        if fi not in ST["dg"]:
            return o_col(s, t, h, sub_dt, depth, fn)
        ST["cur"] = dict(fi=fi, x0=s.x[:s.N].clone())
        ST["ncall"] = ST.get("ncall", {}); ST["ncall"][fi] = ST["ncall"].get(fi, 0) + 1
        try:
            r = o_col(s, t, h, sub_dt, depth, fn)
            try:
                snap(s, "fin", dict(redo=bool(r[0]), K=int(r[1]), tries=[list(q) for q in r[2]]))
            except Exception:
                _err()
            return r
        finally:
            ST["cur"] = None
    WRX.split_col = split_col


def _err():
    import traceback
    with open(os.path.join(ST["out"], "w4_err.txt"), "a") as f_:
        f_.write(traceback.format_exc()[-3000:] + "\n")


def _w27(fx):
    w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], dim=-1)
    o = torch.arange(27, device=fx.device)
    i, j, k = o // 9, (o // 3) % 3, o % 3
    return w[:, 0, i] * w[:, 1, j] * w[:, 2, k]


def _nid(pb, nb1, nb2):
    o = torch.arange(27, device=pb.device)
    i, j, k = o // 9, (o // 3) % 3, o % 3
    return ((pb[:, 0:1] + i) * nb1 + (pb[:, 1:2] + j)) * nb2 + (pb[:, 2:3] + k)


@torch.no_grad()
def snap(s, tag, extra=None):
    C = ST["cur"]
    IP = s.IP.cpu()
    nb1, nb2, nn = int(IP[7]), int(IP[8]), int(IP[9])
    N = int(s.N)
    pb = s.Np.reshape(-1, 3)[:N].long()
    fx = s.fx.reshape(-1, 3)[:N].float()
    h = C["x0"][:, 2].float() - FL_Z
    lay = torch.bucketize(h, torch.tensor(EDG[1:-1], device=h.device))                # 0 .. 15
    sel = torch.arange(0, N, 3, device=h.device)                                      # 입자 셋 중 하나 (층마다 수만 개)
    w = _w27(fx[sel])
    ids = _nid(pb[sel], nb1, nb2).clamp(0, nn - 1)
    L = lay[sel]

    def pz(t3):                                                                        # 노드 [*,3] -> 입자 dz
        return (w * t3.reshape(-1, 3)[:nn, 2][ids]).sum(-1)

    def ps(t1):
        return (w * t1.reshape(-1)[:nn].float()[ids]).sum(-1)
    mq = s.mqn[:nn].float().clamp_min(1e-30)
    F = dict(Ut=pz(s.Ut), yh=pz(s.yh), y0=pz(s.w4y0), fin=pz(s.U), r=ps(s.an[:nn].float() / mq), sk=ps(s.sk))
    if getattr(s, "ykn", None) is not None:
        F["beta"] = ps(torch.pow(2.0, -s.ykn[:nn].float()))
    if ST["t"] and getattr(s, "w4an", None) is not None:
        F["rT"] = ps(s.w4an[:nn] / mq)
    sc = s.w4sc.cpu().numpy()
    nit = int((sc[:, 1] > 0).sum())
    sc = sc[:nit]
    yit = [pz(s.w4yit[i]) for i in range(nit)]
    kit = [ps(torch.pow(2.0, -s.w4kn[i][:nn].float())) for i in range(nit)]
    nop = (F["fin"] - F["y0"]).abs() < 1e-4
    rows = []
    for l_ in range(len(EDG) - 1):
        m = L == l_
        if int(m.sum()) < 20:
            continue
        rw = dict(z=[round(EDG[l_] * 100), round(EDG[l_ + 1] * 100)], n=int(m.sum()), noprog=round(float(nop[m].float().mean()), 4))
        for k_, v in F.items():
            rw[k_] = round(float(v[m].median()), 5)
        rw["yit"] = [round(float(q[m].median()), 5) for q in yit]
        rw["kit"] = [round(float(q[m].median()), 4) for q in kit]
        rows.append(rw)
    rec = dict(f=C["fi"], call=ST["ncall"][C["fi"]], tag=tag, it=nit, g2F2=[(float(a) / float(b)) if b > 0 else None for a, b in sc], layers=rows)
    if extra:
        rec.update(extra)
    ST["log"].write(json.dumps(rec) + "\n")
    ST["log"].flush()
