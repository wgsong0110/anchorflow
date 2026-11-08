"""절 CM45 묶음 escape3: 장벽 표본점 이동을 입자 G2P 와 같은 기저로 (W̄) + 새 실행기 G (노드 질량 가중 G2P). 인자 없으면 이 모듈을 안 부름 = 소스·그래프 그대로.

배경 (절 CM43.1 추정 (b)): 장벽 바닥 표본점 q (팔분 칸 입자 질량 중심 x̄_q, 프레임 시작) 의 걸음 안 위치는
    x_q = x̄_q + Σ_o BW[oc, o] (U_o + thinv (y_o − U_o))        (BW = 팔분 칸 가우스 점 자리 B-스플라인 기저, 칸마다 같은 8 x 27 표)
인데 입자는 자기 자리 기저 ŵ_o(x_p) 로 움직인다 (G 판이면 ŵ_o = w_o(x_p) m_o / Σ w m). 가벼운 노드가 크게 움직이면 둘이 어긋나
장벽은 표본점이 바닥 위라고 보는데 입자는 바닥 밑으로 갈 수 있다.

W̄ (--e3w g): 표본점 이동을 그 팔분 칸 입자들의 G2P 이동의 질량 평균으로 잰다 -- 같은 기저
    W̄_q,o = Σ_{p∈q} m_p ŵ_o(x_p) / Σ_{p∈q} m_p,   x_q = x̄_q + Σ_o W̄_q,o (U_o + thinv (y_o − U_o)) = Σ_p m_p x_p(G2P) / Σ m_p
  g = 1: ŵ = w (원래 G2P), g = 2: ŵ = w m_n / Σ w m_n (G 판 G2P, m_n = 그 프레임 노드 질량 s.mn).
  G2P 가 노드 변위에 선형이므로 걸음 끝 표본점 위치 = 그 칸 입자들의 G2P 끝 위치의 질량 평균 (정확히; 진단 D3 가 확인).
  바꾸는 곳 = 표본점 위치·기울기·헤시안을 쓰는 커널 전부 (몸에 AF_BX·AF_BPH·AF_BACT 가 있고 BW[oc*27+o] 를 쓰는 커널: rt_bar_q·rt_bar_mvq·rt_bar_lsN·
  rt_bar_kfe·rt_bar_y0e·yl_kq·yl_chk·wb_cnt·ul_bar·ws_bk·en_phi_q·rt_bare_q·rt_vres_* 중 소스에 있는 것) 의 BW[oc*27+o] 를 W̄[q] 로.
  띠 고착 (wb_floor_q·rt_floor_q 의 AF_BAR 갈래) 은 위치 판정만 W̄ 로, 고착 질량 몫을 노드에 나누는 무게 (가우스 점 질량 mq 의 기저 BW) 는 그대로
  (노드 질량 m_n = Σ_q m_q BW 와 같은 무게라 고착 몫 an ≤ m_n 유지).
  W̄ 는 프레임마다 한 번 (첫 표본점 커널 앞, 노드 질량이 다 찬 뒤) 입자에서 원자 합으로 모음 (그래프 안). 문턱·클램프·바닥 고정 없음.

G (--e3g 1, 새 실행기): 절 CM43 후보 G (옛 실행기 fem_escape2 --e2g) 와 같은 정의를 새 실행기 G2P 커널 (rt_g2p_ga 탄성 판, 재질 판 aq_g2p·ao_g2p·ao_g2p_a) 에:
    v_p = Σ w m_n v_n / Σ w m_n, Δx_p = Σ w m_n U_n / Σ w m_n, a_p = Σ w m_n a_n / Σ w m_n.  입자 F 누적 (A) 은 그대로 (--fp_interp 가 가우스 점 값으로 덮어씀).
"""
import re

import numpy as np
import torch

G2PN = ("rt_g2p_ga", "aq_g2p", "ao_g2p", "ao_g2p_a")
ST = dict(w=0, g=0, patched=set(), gpatched=set())

E3_SRC = r'''
// ---- 절 CM45 W̄: 표본점 q 의 입자 G2P 기저 질량 평균 W̄_q,o = Σ_p m_p ŵ_o(x_p) / Σ m_p (AF_WQ, 프레임 시작) ----
__global__ void e3_wq(int N, const int* IP, const int* pb, const float* fx, const float* m, const int* pcd, const int* cidx, const float* mn, int g) {
    int p = blockIdx.x*blockDim.x + threadIdx.x;
    if (p >= N || IP[13]) return;
    int c = cidx[pcd[p]];
    int oc = (fx[p*3] >= 1.f ? 4 : 0) + (fx[p*3+1] >= 1.f ? 2 : 0) + (fx[p*3+2] >= 1.f ? 1 : 0);
    size_t q = (size_t)c*8 + oc;
    float w[3][3], dw[3][3];
    for (int a=0;a<3;a++) bwf(fx[p*3+a], w[a], dw[a]);
    int nb1 = IP[7], nb2 = IP[8];
    float ww[27], s = 0.f;
    for (int o=0;o<27;o++) {
        int i = o/9, j = (o/3)%3, k = o%3;
        int nid = ((pb[p*3]+i)*nb1 + (pb[p*3+1]+j))*nb2 + (pb[p*3+2]+k);
        ww[o] = w[0][i]*w[1][j]*w[2][k]*(g == 2 ? mn[nid] : 1.f);
        s += ww[o];
    }
    float sc = m[p]*((g == 2 && s > 0.f) ? 1.f/s : 1.f);                      // G 판 G2P 와 같은 나눔 (Σ w m = 0 이면 G2P 변위 0 -> 몫 0)
    for (int o=0;o<27;o++) atomicAdd(&AF_WQ[q*27+o], ww[o]*sc);
}
__global__ void e3_wqn(const int* IP) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= IP[12]*27) return;
    float bm = AF_BX[(i/27)*4+3];
    AF_WQ[i] = (bm > 0.f) ? AF_WQ[i]/bm : 0.f;
}
'''


def _bodies(src):
    for m in re.finditer(r"__global__\s+void\s+(?:__launch_bounds__\(\d+\)\s+)?(\w+)\s*\(", src):
        b = src.find("{", m.end())
        if b < 0:
            continue
        d, i = 0, b
        while i < len(src):
            if src[i] == "{":
                d += 1
            elif src[i] == "}":
                d -= 1
                if d == 0:
                    break
            i += 1
        yield m.group(1), m.start(), i + 1


def w_apply(src):
    """W̄ 판: 장벽 표본점 커널의 BW[oc*27+o] -> AF_WQ[q*27+o] (띠 고착 커널은 위치 판정만), W̄ 모으는 커널 둘을 덧붙임. 장벽 판 소스 (AF_BX 선언) 가 아니면 그대로."""
    if "__device__ float* AF_BX;" not in src:
        return src
    out, pos, hit = [], 0, []
    for nm, i, j in list(_bodies(src)):
        b = src[i:j]
        if "BW[oc*27+o]" not in b or not any(k_ in b for k_ in ("AF_BX", "AF_BPH", "AF_BACT")) or nm.startswith("e3_"):
            continue
        if nm in ("rt_floor_q", "wb_floor_q"):
            old = "float w = BW[oc*27+o];"
            assert b.count(old) == 1 and b.count("BW[oc*27+o]*mw") == 1, ("--e3w", nm)
            b2 = b.replace(old, "float w = E3_FLW(bm_, q, oc, o);" if nm == "rt_floor_q" else "float w = AF_WQ[(size_t)q*27+o];", 1)
        else:
            qv = "qq" if re.search(r"\bint\s+qq\b", b) else "q"
            assert re.search(r"\bint\s+%s\b" % qv, b) or re.search(r"\b%s\s*=" % qv, b), ("--e3w 표본점 번호", nm)
            b2 = b.replace("BW[oc*27+o]", "AF_WQ[(size_t)%s*27+o]" % qv)
        out.append(src[pos:i] + b2)
        pos = j
        hit.append(nm)
    out.append(src[pos:])
    src = "".join(out)
    src = src.replace('extern "C" {', 'extern "C" {\n__device__ float* AF_WQ;', 1)
    if "rt_floor_q" in hit:                                                              # 매크로는 커널 바로 앞 (AF_BAR 정의 뒤라야 함)
        i = src.index("__global__ void rt_floor_q(")
        src = src[:i] + ("#ifdef AF_BAR\n#define E3_FLW(bm_, q, oc, o) (((bm_) > 0.f) ? AF_WQ[(size_t)(q)*27+(o)] : BW[(oc)*27+(o)])\n"
                         "#else\n#define E3_FLW(bm_, q, oc, o) BW[(oc)*27+(o)]\n#endif\n") + src[i:]
    i = src.rstrip().rfind("}")
    src = src[:i] + E3_SRC + "\n}\n"
    assert "rt_bar_q" in hit, ("--e3w: rt_bar_q 못 고침", hit)
    ST["patched"] |= set(hit)
    return src


def g_apply(src):
    """새 실행기 G: 소스에 있는 G2P 커널 (rt_g2p_ga·aq_g2p·ao_g2p·ao_g2p_a) 몸을 노드 질량 가중 보간으로 (마지막 인자 E3MN = 노드 질량)."""
    from . import fem_escape2 as E2
    hit = []
    if "__global__ void rt_g2p_ga(" in src:
        i = src.index("__global__ void rt_g2p_ga(")
        j = src.index("\n}\n", i) + 3
        b = src[i:j]
        SIG = "float* x, float* v, float* pa, float* F) {"
        ACC = "for (int a=0;a<3;a++) { float uu = U[nid*3+a]; u[a] += ww*uu; vv[a] += ww*vk[nid*3+a]; aa[a] += ww*ak[nid*3+a]; for (int b=0;b<3;b++) A[a*3+b] += uu*gx[b]; }"
        DEC = "    float u[3] = {0,0,0}, vv[3] = {0,0,0}, aa[3] = {0,0,0}, A[9] = {1,0,0, 0,1,0, 0,0,1};\n"
        OUT = "    for (int a=0;a<3;a++) { x[p*3+a] += u[a]; v[p*3+a] = vv[a]; pa[p*3+a] = aa[a]; }"
        for o_ in (SIG, ACC, DEC, OUT):
            assert b.count(o_) == 1, ("--e3g rt_g2p_ga", o_)
        b = b.replace(SIG, SIG[:-3] + ", const float* E3MN) {", 1)
        b = b.replace(DEC, DEC + "    float wm_ = 0.f;                                                             // 절 CM45 G: Σ w m_n\n", 1)
        b = b.replace(ACC, "float wq_ = ww*E3MN[nid]; wm_ += wq_; for (int a=0;a<3;a++) { float uu = U[nid*3+a]; u[a] += wq_*uu; vv[a] += wq_*vk[nid*3+a]; aa[a] += wq_*ak[nid*3+a]; for (int b=0;b<3;b++) A[a*3+b] += uu*gx[b]; }", 1)
        b = b.replace(OUT, "    if (wm_ > 0.f) for (int a=0;a<3;a++) { u[a] /= wm_; vv[a] /= wm_; aa[a] /= wm_; }   // 절 CM45 G\n" + OUT, 1)
        src = src[:i] + b + src[j:]
        hit.append("rt_g2p_ga")
    if "__global__ void aq_g2p(" in src and "__global__ void ao_g2p_a(" in src:
        src = E2.g_apply(src)                                                           # 절 CM43 G 글자 그대로 (aq_g2p·ao_g2p_a, 마지막 인자 E2MN)
        hit += ["aq_g2p", "ao_g2p_a"]
    if "__global__ void ao_g2p(" in src:                                                 # 모래 재질 판 (aq_g2p 와 같은 누적·출력 줄)
        i = src.index("__global__ void ao_g2p(")
        j = src.index("\n}\n", i) + 3
        b = src[i:j]
        SIG = "float* x, float* v, float* F, int* cnt) {"
        for o_ in (SIG, E2.G_ACC, E2.G_OUT, "    int nrm = 0;\n"):
            assert b.count(o_) == 1, ("--e3g ao_g2p", o_)
        b = b.replace(SIG, SIG[:-3] + ", const float* E3MN) {", 1)
        b = b.replace("    int nrm = 0;\n", "    int nrm = 0;\n    float wm = 0.f;                                                              // 절 CM45 G: Σ w m_n\n", 1)
        b = b.replace(E2.G_ACC, "if (last) { float wq = ww*E3MN[nid]; if (a == 0) wm += wq; u[a] += wq*uu; vv[a] += wq*vk[nid*3+a]; } }", 1)
        b = b.replace(E2.G_OUT, "    if (wm > 0.f) for (int a=0;a<3;a++) { u[a] /= wm; vv[a] /= wm; }               // 절 CM45 G\n" + E2.G_OUT, 1)
        src = src[:i] + b + src[j:]
        hit.append("ao_g2p")
    ST["gpatched"] |= set(hit)
    return src


def _bind(s):
    """AF_WQ 장치 전역을 이 풀이기의 모듈에 (다시 할당·모듈 바꿈 뒤마다)."""
    if getattr(s, "e3wq", None) is None:
        return
    from . import fem_an_rt as ART
    assert getattr(s, "pkmod", None) is None, "--e3w: 지속형 판 모듈은 안 고침"
    try:
        g = s.mod.get_global("AF_WQ")
    except Exception:
        return                                                                           # 고치지 않은 모듈 (장벽 없는 판의 첫 모듈 등)
    ART.cp.ndarray((1,), dtype=ART.cp.uint64, memptr=g)[0] = int(s.e3wq.data_ptr())
    s._e3bound = (id(s.mod), int(s.e3wq.data_ptr()))


def install(w=0, g=0, newg=False):
    """w: 0 끔 / 1 W̄ 원래 G2P 기저 / 2 W̄ G 판 기저 (노드 질량 가중). newg: 새 실행기 G2P 를 G 로."""
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    P = ART.P
    ST["w"], ST["g"] = int(w), bool(newg)
    _obs = ART.build_src
    _oas = AORT._src

    def bs(*a, **kw):
        r = _obs(*a, **kw)
        if ST["g"]:
            r = g_apply(r)
        return w_apply(r) if ST["w"] else r
    ART.build_src = bs
    if ST["w"]:
        def as_(*a, **kw):
            return w_apply(_oas(*a, **kw))
        AORT._src = as_
    from . import fem_an_rt_v0 as V0                                                   # 옛 실행기 (수박) 의 바탕 풀이기 (fem_ao_rt -> fem_det_rt -> fem_an_rt_v0.RTSolver)

    def wrap_k(cls):
        _ok = cls.k

        def k(s, name, n, args, b=128):
            if ST["g"] and name in G2PN and name in ST["gpatched"]:
                args = tuple(args) + (P(s.mn),)
            elif ST["g"] and name in ("rt_g2p_endg",):
                raise RuntimeError("--e3g: rt_g2p_endg (generalized-α 아닌 판) 는 안 고침")
            if ST["w"] and not getattr(s, "_e3in", False):
                if name == "rt_reset":
                    hasbar = bool(getattr(s, "bar", False)) or getattr(s, "wbar", None) is not None
                    assert not hasbar or "rt_bar_q" in ST["patched"], "--e3w: 장벽 판인데 장벽 커널을 못 고침"
                    if not hasbar and not ST.get("nobar_said"):
                        ST["nobar_said"] = True
                        print("[절 CM45] 장벽 바닥 없는 씬 -- W̄ 해당 없음 (표본점 없음, 소스 그대로)", flush=True)
                    s._e3done = not hasbar
                elif name in ST["patched"] and not getattr(s, "_e3done", True):
                    assert getattr(s, "_e3bound", None) == (id(s.mod), int(s.e3wq.data_ptr())), "--e3w: AF_WQ 가 이 모듈·버퍼에 안 묶임"
                    assert not getattr(s, "det", False), "--e3w: 결정적 판 (AF_BDET 노드 모으기 커널) 은 안 고침"
                    s._e3in = True
                    try:
                        i32 = np.int32
                        s.zf(s.e3wq, 12, 27)
                        _ok(s, "e3_wq", s.N, (i32(s.N), P(s.IP), P(s.Np), P(s.fx), P(s.m), P(s.pcd), P(s.cidx), P(s.mn), i32(ST["w"])))
                        _ok(s, "e3_wqn", s.capQc * 8 * 27, (P(s.IP),), 256)
                    finally:
                        s._e3in = False
                    s._e3done = True
                    s._e3trig = (name, int(getattr(s, "_e3trig", (None, 0))[1]) + 1)          # 진단: 이 풀이기에서 W̄ 를 모은 횟수·마지막으로 앞선 커널
            return _ok(s, name, n, args, b)
        cls.k = k

    def wrap_alloc(cls):
        _oa = cls.alloc

        def alloc(s, need):
            r = _oa(s, need)
            s.e3wq = torch.zeros(s.capQc * 8 * 27, device=s.dev, dtype=torch.float32)
            _bind(s)
            return r
        cls.alloc = alloc
    for cls in (ART.RTSolver, V0.RTSolver):
        wrap_k(cls)
    if not ST["w"]:
        return
    for cls in (ART.RTSolver, V0.RTSolver):
        wrap_alloc(cls)
    _ob = ART.RTSolver.oct_bind

    def oct_bind(s):
        r = _ob(s)
        _bind(s)
        return r
    ART.RTSolver.oct_bind = oct_bind
    _oba = AORT.AORTSolver.bind_act

    def bind_act(s):
        r = _oba(s)
        _bind(s)
        return r
    AORT.AORTSolver.bind_act = bind_act
