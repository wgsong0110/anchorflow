"""절 CM44 수박 깨짐 모양 3 -- 늦은 손상 기어오름 (덜 풀린 걸음의 가우스 점 부피 흔들림을 커밋 RM 이 logJp 로 쌓음, 절 CM41) 막기 후보 (수박 옛 실행기 exe/fem_ao_rt_sim.py 만).
플래그를 안 주면 이 모듈을 안 부름 (생성 소스·커널 열·프레임 경로 그대로). 소스 바꾸기는 fem_wmfrac.apply (+ 절 CM41 감싸기) 결과에 글자 치환으로만, 새 커널 없음.

걸음 수렴 판정 (새 문턱 없음, 기존 판정만):
  --ws3 1 = 허용치 εN (--tolN): 그 걸음 뉴턴이 rt_ctl op 3 의 ‖g‖² ≤ εN² F² (또는 첫 반복 ‖g‖ = 0) 로 멈춤 -> AF_W3EN = 1 (op 1 걸음 시작에 0).
  --ws3 2 = 절 CM24 덜 풀린 걸음 판정 (uf_end, --uf_res 판: 허용치로 안 멈췄고 (보폭 최소 < 1 또는 마지막 반복 시작 ‖g‖ > F)) 의 반대 -> AF_W3OK = 1 − ufl[k].
후보 (커밋 aq_commit 에서, 수렴 안 한 걸음만):
  기본 (--ws3_lj 0) 소성 미룸: RM 을 안 하고 바로 돌아감 -- F_q (프레임 시작 기준 F) · 상태 (ys, logJp, al) 그대로. 그 걸음 변형은 탄성 시험값으로 남고
      (프레임 끝 fpi_q 가 (I + ∇U) F_q 를 입자로), 다음에 수렴한 걸음의 커밋이 쌓인 시험값 전체를 한 번에 RM (가는 흔들림·오는 흔들림이 서로 지움).
      뉴턴 안 힘 (aq_elem) 은 지금처럼 매 반복 시험 F 를 RM 한 응력 -- 힘 경로 그대로.
  --ws3_lj 1 손상 변수 고정: RM 은 그대로 해서 F_q 는 항복면 위로 투영 (응력 경로 그대로), 상태 (ys, logJp, al) 만 걸음 시작 값 유지.
  --ws3_win 1: 절 CM41 충격 창 (--ws2_win) 이 끝난 프레임 (fem_wmshape2.ST["sw"]) 다음부터만 켬 (그 전은 지금 그대로).
(2) 첫 접촉 후보 --ws3_ep 1 (--ws2_win 위): 충격 창이 끝날 때까지 (f1 ~ 창 끝 프레임) 뉴턴 안 힘·접선은 탄성 (prm[15] = 1, aq_elem 의 RM 끔) 이고
  커밋 RM 만 함 (탄성 예측 -> 걸음 끝 소성 보정, 등분 --rm_sub 그대로) -- 창 끝 다음 프레임부터 prm[15] = 0 (지금 그대로). 커밋 RM 사본 (w2_rms_c) 만 prm[15] 문턱을 1.5 로.
--ws3_pf 1 (--ws3 위): 후보가 켜진 프레임에서 마지막 걸음이 수렴 안 해 손상 변수를 확정 안 했으면 입자 logJp 도 프레임 시작 값 그대로
  (프레임 끝 보간 교체 fpi_p 가 덮은 logJp 를 되돌림; F 는 보간 교체 그대로) -- 매 프레임 보간·다시 모으기가 손상을 이웃으로 번지게 하는 것 막기.
  커밋이 AF_W3LAST (마지막 걸음 수렴 여부) 를 적고, aq_p2g_s 앞 w3_lo (입자 logJp 사본) · fpi_p 뒤 w3_pf (되돌림) 두 커널 (AF_W3PL = 후보 켜짐).
진단 (결과 불변): 프레임마다 커밋한 걸음 · 미룬 (또는 고정한) 걸음 수 (AF_W3CNT, 다시 하기 시도 포함) -> fem.jsonl ws3.
"""
import numpy as np

DECL = "\n__device__ int AF_W3PL;\n__device__ int AF_W3LAST;\n__device__ int AF_W3MODE;\n__device__ int AF_W3EN;\n__device__ int AF_W3OK;\n__device__ int AF_W3CNT[4];"
CTL_OP1_OLD = "if (op == 1) { CT[0] = 0; CT[1] = 0; CT[6]++; AF_SKIP = 0; return; }"
CTL_OP1_NEW = "if (op == 1) { CT[0] = 0; CT[1] = 0; CT[6]++; AF_SKIP = 0; AF_W3EN = 0; return; }"
CTL_Z_OLD = "if (g2 == 0.0) { CT[0] = 1; AF_SKIP = 1; return; } }"
CTL_Z_NEW = "if (g2 == 0.0) { CT[0] = 1; AF_SKIP = 1; AF_W3EN = 1; return; } }"
CTL_N_OLD = "else if (eN2 >= 0.0 && g2 <= eN2*CD[2]) { CT[0] = 1; AF_SKIP = 1; return; }"
CTL_N_NEW = "else if (eN2 >= 0.0 && g2 <= eN2*CD[2]) { CT[0] = 1; AF_SKIP = 1; AF_W3EN = 1; return; }"
UF_OLD = "    ufl[k] = ((adapt ? (CT[0] == 0) : 1) && (cut || big)) ? 1 : 0; }"
UF_NEW = "    ufl[k] = ((adapt ? (CT[0] == 0) : 1) && (cut || big)) ? 1 : 0; AF_W3OK = 1 - ufl[k]; }"
CM_HEAD = "__global__ void aq_commit("
CM_Q_OLD = "    if (q >= IP[12]) return;\n"
CM_Q_NEW = ("    if (q >= IP[12]) return;\n"
            "    int w3c_ = ((AF_W3MODE & 3) == 1) ? AF_W3EN : (((AF_W3MODE & 3) == 2) ? AF_W3OK : 1);   // 절 CM44: 이 걸음 수렴 (기존 판정)\n"
            "    if (q == 0 && (AF_W3MODE & 3)) { atomicAdd(&AF_W3CNT[w3c_ ? 0 : 1], 1); AF_W3LAST = w3c_; }\n"
            "    if (!w3c_ && !(AF_W3MODE & 4)) return;                                          // 절 CM44 미룸: 수렴 안 한 걸음은 소성 상태 확정 안 함\n")
CM_S_OLD = "    Sq[(size_t)q*4] = ys; Sq[(size_t)q*4+1] = lj; Sq[(size_t)q*4+2] = al;\n"
CM_S_NEW = "    if (w3c_) { Sq[(size_t)q*4] = ys; Sq[(size_t)q*4+1] = lj; Sq[(size_t)q*4+2] = al; }      // 절 CM44 --ws3_lj: 수렴 안 한 걸음은 손상 변수 그대로\n"


def _rep(s, o, n, cnt=1):
    assert s.count(o) == cnt, (s.count(o), o[:120])
    return s.replace(o, n)


EP_OLD = "bool ch = (prm[15] > 0.5f) ? false : pl_rm(mat, se, prm, ys, lj, al, dt);"
EP_NEW = "bool ch = (prm[15] > 1.5f) ? false : pl_rm(mat, se, prm, ys, lj, al, dt);   // 절 CM44 --ws3_ep: 커밋 RM 은 prm[15] = 1 (힘만 탄성) 에도 함"


PF_SRC = r"""
// ---- 절 CM44 --ws3_pf: 손상 변수를 확정 안 한 프레임은 입자 logJp 도 그대로 ----
__global__ void w3_lo(int N, const int* IP, const float* ljp, float* lo) { int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= N || IP[13]) return; lo[p] = ljp[p]; }
__global__ void w3_pf(int N, const int* IP, const float* lo, float* ljp) { int p = blockIdx.x*blockDim.x + threadIdx.x; if (p >= N || IP[13]) return; if (AF_W3PL && !AF_W3LAST) ljp[p] = lo[p]; }
"""


def src_apply(r, ep=0, pf=0):
    if ep:
        i = r.index("__device__ bool w2_rms_c(")
        j = r.index("\n}\n", i) + 3
        r = r[:i] + _rep(r[i:j], EP_OLD, EP_NEW) + r[j:]
    assert "__global__ void rt_ctl(" in r and "__global__ void uf_end(" in r, "절 CM44 는 허용치 판 (--tolN) + 덜 풀림 판정 (--uf_res) 위"
    r = r.replace("extern \"C\" {", "extern \"C\" {" + DECL, 1)
    r = _rep(r, CTL_OP1_OLD, CTL_OP1_NEW)
    r = _rep(r, CTL_Z_OLD, CTL_Z_NEW)
    r = _rep(r, CTL_N_OLD, CTL_N_NEW)
    r = _rep(r, UF_OLD, UF_NEW)
    k = r.index(CM_HEAD)
    e = r.find("__global__", k + 10)
    e = len(r) if e < 0 else e
    body = _rep(r[k:e], CM_Q_OLD, CM_Q_NEW)
    body = _rep(body, CM_S_OLD, CM_S_NEW)
    r = r[:k] + body + r[e:]
    if pf:
        i = r.rstrip().rfind("}")
        r = r[:i] + PF_SRC + "\n}\n"
    return r


CFG = dict(mode=0, lj=0, win=0, ep=0, pf=0)
ST = dict(S=None, on=False, at=None)


def install(a):
    from . import fem_ao_rt as AORT
    from . import fem_wmfrac as WF
    CFG.update(mode=int(a.ws3), lj=int(a.ws3_lj), win=int(a.ws3_win), ep=int(a.ws3_ep), pf=int(a.ws3_pf))
    assert not CFG["pf"] or CFG["mode"], "--ws3_pf 는 --ws3 위"
    assert CFG["mode"] in (0, 1, 2) and (CFG["mode"] or CFG["ep"])
    if getattr(AORT, "_ws3", False):
        return
    AORT._ws3 = True
    o_apply = WF.apply

    def apply(r, wf):
        return src_apply(o_apply(r, wf), CFG["ep"], CFG["pf"])
    WF.apply = apply
    o_bind = AORT.AORTSolver.bind_act

    def bind_act(s):
        o_bind(s)
        _set(s)
    AORT.AORTSolver.bind_act = bind_act
    if CFG["pf"]:
        import torch
        o_alloc = AORT.AORTSolver.alloc

        def alloc(s, need):
            o_alloc(s, need)
            s.w3lo = torch.zeros(s.N, device=s.dev, dtype=torch.float32)
        AORT.AORTSolver.alloc = alloc
        ok = AORT.AORTSolver.k
        P = AORT.ART.P
        i32 = np.int32

        def k(s, name, n, args, b=128):
            if name == "aq_p2g_s":
                ok(s, "w3_lo", s.N, (i32(s.N), P(s.IP), P(s.ljp), P(s.w3lo)), 128)
            r = ok(s, name, n, args, b)
            if name == "fpi_p":
                ok(s, "w3_pf", s.N, (i32(s.N), P(s.IP), P(s.w3lo), P(s.ljp)), 128)
            return r
        AORT.AORTSolver.k = k


def _g(s, nm, n=1):
    from . import fem_ao_rt as AORT
    return AORT.ART.cp.ndarray((n,), dtype=AORT.ART.cp.int32, memptr=s.mod.get_global(nm))


def _set(s):
    if getattr(s, "mod", None) is None:
        return
    on = ST["on"] or not CFG["win"]
    _g(s, "AF_W3MODE")[0] = (CFG["mode"] | (4 if CFG["lj"] else 0)) if (on and CFG["mode"]) else 0
    if CFG["pf"]:
        _g(s, "AF_W3PL")[0] = 1 if on else 0


def setup(S):
    assert S.ufj and S.adapt, "절 CM44 는 --tolN 허용치 + --uf_res 판정 위"
    ST["S"] = S
    _set(S)
    if CFG["ep"]:
        S.prm[15] = 1.0                                                                # 절 CM44 --ws3_ep: 충격 창 끝까지 뉴턴 힘은 탄성


def frame(fi):
    """프레임 끝 (읽기만 + 창 뒤 켜기): 커밋·미룬 걸음 수 (다시 하기 시도 포함)."""
    S = ST["S"]
    c = _g(S, "AF_W3CNT", 4)
    v = [int(q) for q in c.get()]
    c[:] = 0
    o = dict(on=bool(ST["on"] or not CFG["win"]), ok=v[0], defer=v[1], ufl=[int(q) for q in S.ufl[:S.K].cpu()])
    from . import fem_wmshape2 as WS2
    if CFG["ep"] and float(S.prm[15]) > 0.5 and WS2.ST.get("sw") is not None:
        S.prm[15] = 0.0
        o["ep_end"] = fi + 2
        print(f"[절 CM44] 충격 창 끝 (f{WS2.ST['sw']}) -> f{fi + 2} 부터 뉴턴 힘 RM 다시 켬", flush=True)
    if CFG["win"] and not ST["on"]:
        if WS2.ST.get("sw") is not None:
            ST["on"], ST["at"] = True, fi + 2
            _set(S)
            o["start"] = fi + 2
            print(f"[절 CM44] 충격 창 끝 (f{WS2.ST['sw']}) -> f{fi + 2} 부터 수렴 안 한 걸음 {'손상 변수 고정' if CFG['lj'] else '소성 미룸'}", flush=True)
    return o
