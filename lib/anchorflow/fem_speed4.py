"""절 CM50 묶음 speed4 (조정자 위임 11-08, 문제 4 수박 60 fps): 공통 후보 F 수박 -- 단계별 시간 기록 (결과 불변) + 결과 불변 가속.
감싸개 exe/speed4_wrap.py 가 부름 (인자 없으면 이 모듈을 안 부름 = 소스·그래프 그대로). 옛 실행기 (fem_ao_rt.AORTSolver, 수박) 만 건드림 -- 새 실행기 (레고·wolf·ficus·bread) 는 안 건드림.

(P) 단계 시간 기록 (--s4p 1, 결과 불변): 그래프 캡처 때만, 단계가 바뀌는 커널 앞에 한 스레드 커널 sp_st (globaltimer 를 따로 둔 버퍼 SP_T·SP_C 에 씀) 를 넣음.
    재생 뒤 (프레임마다) 이웃 표식 사이 시간을 앞 표식의 단계에 더함 -- 실제 그래프 재생 (나눈 그래프·예산·다시 하기 포함) 의 단계별 GPU 시간.
    단계: P2G (프레임 시작 ~ 걸음 시작 앞: 칸·상자·P2G·가우스 점·노드 속도), 걸음 시작 (관성 예측·띠 고착·시작점·장벽 시작 판정·y0loc·시작 뒤집힘 필터),
    W̄ (e3_*), R2·A2 (e5_*), 뉴턴 반복 안 [요소 (aq_elem) · 장벽 기울기 (rt_bar_q·rt_bar_n) · 기울기·판정 (rt_grad·uf_g·en_gate) · PCG (헤시안 곱·장벽 곱·CG 벡터)
    · 장벽·뒤집힘 보폭 (ws_*·aw_*·rt_ytry·rt_bar_kfe·rt_lsup) · 선탐색 에너지 (en_*)], 걸음 끝 (uf_end·ul_*·ao_stepend), RM (aq_commit), G2P (ao_g2p_a·aq_g2p·fpi_*),
    구간 사이 (나눈 그래프 경계 -> 다음 구간 첫 커널: 호스트 읽기·그래프 띄우기), 시도 사이 (쪼개기·X3c 다시 하기 시도 경계, GPU 시간 아님).
    0 채우기·제어 커널 (rt_zero_*·rt_ctl·rt_sc_clear) 은 앞 단계에 붙음. 다른 모듈이 s.k 를 거치지 않고 띄우는 작은 커널 (시간 예산 tb_*) 은 앞 단계에 붙음.
    표식 커널 하나 ~1-3 µs 라 기록 판의 프레임 시간은 조금 늘어남 (기록 없는 단독 판과 따로 잼).

(S1) 장벽·뒤집힘 국소 보폭 (절 CM30 (b) fem_wmsparse.newton) 의 확인 라운드 조기 끝 (--s4w 1, 결과 불변):
    라운드 r (sc 1) 이 어긴 점 0 (fl[0] + fl[1] = 0) 이면 kn2 에 아무것도 안 흩고 ws_merge 도 kn 을 안 바꿈 -> 다음 라운드·최종 확인 (sc 0) 은 같은 kn·같은 입력이라
    후보 y1·방향 dd·kq·kk·어긴 수 모두 같고 (원자 연산은 정수 최대·합이라 순서 무관) fl 은 0 그대로 -> 그 뒤 ws_cand·ws_inv_q·ws_nk·ws_bk·ws_merge 를 건너뛰어도
    kn·wsy1·wsdd·wskq·wskk·wsfl (= ws_fb·ws_lsup 입력) 이 비트 같음. 장치 표시 WS4_DN (ws_init 이 0 으로, 라운드 끝 ws4_chk 가 어긴 점 0 이면 1), 다섯 커널 첫머리 'WS4_DN 이면 돌아감'.
    --s4w 2 = 확인 판: 건너뛰지 않고, WS4_DN 이 선 뒤 라운드에서 어긴 점이 나오면 WS4_BAD 를 셈 (0 이어야 위 논리가 맞음).
"""
import json
import os
import re

import numpy as np
import torch

# ======================= (P) 단계 시간 기록 =======================
SP_SRC = r"""
extern "C" {
#define SPN 65536
__device__ unsigned long long SP_T[SPN];
__device__ int SP_C[SPN];
__device__ int SP_N;
__global__ void sp_st(int c) {
    int i = SP_N;
    if (i < SPN) { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); SP_T[i] = t; SP_C[i] = c; }
    SP_N = i + 1;
}
}
"""
SPN = 65536
CATS = ["P2G", "걸음 시작", "W̄", "R2·A2", "요소", "장벽 기울기", "기울기·판정", "PCG", "장벽·뒤집힘 보폭", "선탐색 에너지", "걸음 끝", "RM", "G2P", "구간 사이", "시도 사이", "기타"]
CI = {c: i for i, c in enumerate(CATS)}
NEWTON_C = {CI[c] for c in ("요소", "장벽 기울기", "기울기·판정", "PCG", "장벽·뒤집힘 보폭", "선탐색 에너지")}
BEGIN = 1 << 14
INHERIT = {"rt_zero_i", "rt_zero_f", "rt_zero_d", "rt_ctl", "rt_sc_clear", "rt_reset"}
N_ELEM = {"aq_elem", "ao_elem", "ao_elem_s"}
N_BAR = {"rt_bar_q", "rt_bar_n", "wb_cnt"}
N_GRAD = {"rt_grad", "rt_grad_c", "uf_g", "en_gate", "c2_gate"}
N_PCG = {"rt_bar_mv", "rt_ap", "rt_ap_c", "rt_cg1", "rt_cg1_c", "rt_cg2", "rt_cg2_c", "rt_sc_roll", "rt_cg1m", "rt_dotz", "rt_pinit", "ao_warm"}
N_STEP = {"rt_ytry", "rt_bar_kfe", "wb_kmin", "rt_lsup", "rt_lsup_c", "uf_min", "ao_tvec", "aw_tdir", "ws4_chk"}
T_RM = {"aq_commit", "ao_commit"}
T_G2P = {"ao_g2p_a", "aq_g2p", "ao_g2p", "fpi_q", "fpi_p", "lji_p", "pf_g2p"}


class SP:
    mod = None
    fn = None
    on = False
    unk = set()


def _cat(s, name):
    """단계 번호 (None = 앞 단계에 붙음)."""
    if name in INHERIT:
        return None
    if name.startswith("e3_"):
        return CI["W̄"]
    if name.startswith("e5_"):
        return CI["R2·A2"]
    ph = getattr(s, "_spph", "pre")
    if ph == "pre":
        return CI["P2G"]
    if ph == "start":
        return CI["걸음 시작"]
    if ph == "newton":
        if name in N_ELEM:
            return CI["요소"]
        if name in N_BAR:
            return CI["장벽 기울기"]
        if name in N_GRAD:
            return CI["기울기·판정"]
        if name in N_PCG or name.startswith("ao_mv") or name.startswith("rt_mg_"):
            return CI["PCG"]
        if name in N_STEP or name.startswith("ws_") or name.startswith("aw_") or name.startswith("s3_"):
            return CI["장벽·뒤집힘 보폭"]
        if name.startswith("en_"):
            return CI["선탐색 에너지"]
        SP.unk.add(("newton", name))
        return CI["기타"]
    if name in T_RM:
        return CI["RM"]
    if name in T_G2P:
        return CI["G2P"]
    return CI["걸음 끝"]


def _stamp(code):
    from . import fem_rt_cuda as RTC
    with RTC._Stream():
        SP.fn((1,), (1,), (np.int32(code),))


def _cap():
    return bool(torch.cuda.is_current_stream_capturing())


def _emit(s, cid, begin=False):
    it = int(getattr(s, "_spit", 0)) if cid in NEWTON_C else 0
    ks = max(0, int(getattr(s, "_spks", 0)))
    code = cid + 32 * min(it, 15) + 512 * min(ks, 7) + (BEGIN if begin else 0)
    if code != getattr(s, "_splast", None) or begin:
        _stamp(code)
        s._splast = code


def sp_g(name, dt, n=1):
    import cupy as cp
    return cp.ndarray((n,), dtype=dt, memptr=SP.mod.get_global(name))


def sp_frame():
    """이 프레임 (모든 시도) 표식 -> 시도별·단계별 GPU ms."""
    torch.cuda.synchronize()
    n = int(sp_g("SP_N", np.int32)[0])
    m = min(n, SPN)
    if m == 0:
        return dict(n=0)
    T = sp_g("SP_T", np.uint64, SPN)[:m].get().astype(np.int64)
    C = sp_g("SP_C", np.int32, SPN)[:m].get()
    tries, cur = [], None
    for i in range(m):
        c = int(C[i])
        if c & BEGIN:
            cur = dict(cat={}, it={}, t0=int(T[i]), t1=None)
            tries.append(cur)
        if cur is None:
            continue
        cid = c & 31
        if cid == CI["시도 사이"]:
            cur["t1"] = int(T[i])
        if i + 1 < m:
            d = (int(T[i + 1]) - int(T[i])) * 1e-6
            nm = CATS[cid]
            cur["cat"][nm] = cur["cat"].get(nm, 0.0) + d
            if cid in NEWTON_C:
                it = (c >> 5) & 15
                cur["it"][it] = cur["it"].get(it, 0.0) + d
    out = []
    for q in tries:
        gpu = ((q["t1"] - q["t0"]) * 1e-6) if q["t1"] else None
        out.append(dict(ms=round(gpu, 4) if gpu is not None else None, cat={k: round(v, 4) for k, v in q["cat"].items() if k != "시도 사이"},
                        gap=round(q["cat"].get("시도 사이", 0.0), 4), it={str(k): round(v, 4) for k, v in sorted(q["it"].items())}))
    sp_g("SP_N", np.int32)[0] = 0
    torch.cuda.synchronize()
    return dict(n=n, tries=out)


def sp_install():
    import cupy as cp
    from . import fem_ao_rt as AORT
    from . import fem_common3 as C3
    SP.mod = cp.RawModule(code=SP_SRC)
    SP.fn = SP.mod.get_function("sp_st")
    SP.on = True
    from . import fem_an_rt_v0 as V0
    cls = AORT.AORTSolver
    o_rec, o_ctl = cls.record, cls.ctl
    o_k = V0.RTSolver.k                                                                # 가장 안쪽 (다른 모듈이 V0.RTSolver.k 를 감싸 띄우는 e3_·e5_ 커널도 지나감)

    def record(s, t_frame, K=None):
        s._spph, s._spit, s._spks, s._splast = "pre", 0, -1, None
        cap = _cap()
        if cap:
            _emit(s, CI["P2G"], begin=True)
        r = o_rec(s, t_frame, K)
        if cap:
            _stamp(CI["시도 사이"])
            s._splast = None
        return r

    def k(s, name, n, args, b=128):
        if isinstance(s, cls):
            if name in ("ao_step0_ga", "rt_step0", "ao_step0_th"):
                s._spph, s._spit = "start", 0
                s._spks = int(getattr(s, "_spks", -1)) + 1
            if _cap():
                c = _cat(s, name)
                if c is not None:
                    _emit(s, c)
        return o_k(s, name, n, args, b)

    def ctl(s, op, it=0):
        if op == 2:
            s._spph, s._spit = "newton", int(it)
        elif op == 8:
            s._spph = "tail"
        return o_ctl(s, op, it)
    cls.record, cls.ctl = record, ctl
    V0.RTSolver.k = k
    o_mark = C3.mark

    def mark(s, kind):
        if getattr(s, "_c3cap", None) is not None and isinstance(s, cls):
            _stamp(CI["구간 사이"])
            s._splast = None
        return o_mark(s, kind)
    C3.mark = mark


# ======================= (S1) 국소 보폭 확인 라운드 조기 끝 =======================
WS4_HEAD = "__device__ int WS4_DN; __device__ unsigned long long WS4_BAD, WS4_NCLEAN, WS4_NROUND;\n"
WS4_K = r"""
__global__ void ws4_chk(const int* fl) {
    if (AF_SKIP) return;
    WS4_NROUND++;
    if (!WS4_DN && fl[0] + fl[1] == 0) { WS4_DN = 1; WS4_NCLEAN++; }
}
"""
WS4_SKIPK = ("ws_cand", "ws_inv_q", "ws_nk", "ws_bk", "ws_merge")


class WS4:
    mode = 0


def _ins_body(src, name, text):
    m = re.search(r"__global__\s+void\s+(?:__launch_bounds__\(\d+\)\s+)?" + name + r"\s*\(", src)
    assert m is not None, f"--s4w: {name} 없음"
    b = src.index("{", m.end())
    assert len(re.findall(r"__global__\s+void\s+(?:__launch_bounds__\(\d+\)\s+)?" + name + r"\s*\(", src)) == 1, name
    return src[:b + 1] + text + src[b + 1:]


def ws4_src(src):
    if "ws_init(" not in src:
        return src
    assert "__device__ int AF_SKIP;" in src, "--s4w 는 허용치 판 (AF_SKIP) 위"
    src = src.replace('extern "C" {', 'extern "C" {\n' + WS4_HEAD, 1)
    src = _ins_body(src, "ws_init", "\n    if (blockIdx.x == 0 && threadIdx.x == 0) WS4_DN = 0;                               // 절 CM50 S1\n")
    if WS4.mode == 1:
        for nm in WS4_SKIPK:
            src = _ins_body(src, nm, " if (WS4_DN) return;")
    else:
        for nm, o_ in (("ws_bk", "atomicAdd(&fl[0], 1);"), ("ws_nk", "atomicAdd(&fl[1], 1);")):
            m = re.search(r"__global__\s+void\s+(?:__launch_bounds__\(\d+\)\s+)?" + nm + r"\s*\(", src)
            j = src.index("\n}\n", m.end())
            body = src[m.start():j]
            assert body.count(o_) == 1, (nm, o_)
            src = src[:m.start()] + body.replace(o_, o_ + " if (WS4_DN) atomicAdd(&WS4_BAD, 1ull);", 1) + src[j:]
    i = src.rfind("}")
    return src[:i] + WS4_K + src[i:]


def ws4_install(mode):
    from . import fem_ao_rt as AORT
    from .fem_ao_rt import P
    WS4.mode = int(mode)
    cls = AORT.AORTSolver
    o_init, o_k = cls.__init__, cls.k

    def __init__(s, *a, **kw):
        if not getattr(AORT, "_ws4_src", False):                                      # 처음 풀이기 만들 때 바깥에서 감쌈 (다른 모듈의 소스 바꾸기가 모두 끝난 뒤)
            AORT._ws4_src = True
            o_s = AORT._src
            AORT._src = lambda *a_, **k_: ws4_src(o_s(*a_, **k_))
        return o_init(s, *a, **kw)

    def k(s, name, n, args, b=128):
        r = o_k(s, name, n, args, b)
        if name == "ws_bk" and int(args[-1]) == 1:                                      # 라운드 (sc 1) 끝: 어긴 점 0 이면 표시
            o_k(s, "ws4_chk", 1, (args[-2],), 1)
        return r
    cls.__init__, cls.k = __init__, k


def ws4_counts(s):
    import cupy as cp
    g = lambda nm: cp.ndarray((1,), dtype=cp.uint64, memptr=s.mod.get_global(nm))
    o = {}
    for nm, key in (("WS4_NROUND", "round"), ("WS4_NCLEAN", "clean"), ("WS4_BAD", "bad")):
        a = g(nm)
        o[key] = int(a[0])
        a[0] = 0
    return o


# ======================= (X1·X2) 버릴 시도의 꼬리 건너뛰기 · X3c 다시 하기를 걸음 시작부터 (결과 불변) + (X3e4) 후보 =======================
# 수박 F 의 K 1 시도는 무너진 걸음 (덜 풀림 ∧ !(‖g‖²/F² ≤ --ue_bigT), C3.nfail) 이 있으면 버려짐: X3c 첫 시도 (AF_WRXON 0, --wrx_col 1) 는 입자 상태를 되돌리고
#   켠 채 다시, 쪼개기 (split_step depth > 0) 는 되돌리고 K 2 로 다시. 판정 값 ufl·ufg 는 걸음 끝 uf_end 가 쓰고 그 뒤 커널 (걸음 끝 갱신·RM 커밋·G2P·R2 끝 기준 높이)
#   은 입자·노드 상태만 바꿈 -> 되돌리는 시도에서는 버려짐.
# X1 (--s4x 1): K 1 그래프를 uf_end 뒤에서 한 번 더 나누고 ('x' 구간), 재생 때 '버려질 시도' 이면 그 경계에서 호스트가 넘침 (IP[13]) 0 · nfail > 0 을 읽어
#   나머지 구간 (꼬리) 을 안 띄움. 판정·되돌림·다음 시도는 그대로 (같은 ufl·ufg 를 split_step 이 다시 읽음). 시간 예산 기록 tb_end 도 꼬리라 그 시도는 C_tail 표본을 안 냄.
# X2 (--s4x 2, X1 포함): X3c 다시 하기 (켠 채 K 1) 를 프레임 처음 (P2G) 부터가 아니라 걸음 시작 ('s' 구간, ao_step0_ga 앞) 부터 같은 재생 안에서.
#   근거: 걸음 시작 ~ uf_end 커널은 P2G 단계 산물 (U·vk·gaA·mn·mqn·Fq·Vq·mq·Sq·W̄ 재료·칸 표) 을 읽기만 하고 (aq_elem·en_phi_q 는 Fq·Sq const, U·vk·gaA 는 꼬리
#   ao_stepend·aq_commit 만 씀 -- X1 로 꼬리를 안 띄움), 걸음 안에서 쓰는 노드 상태 (Ut·z0·y·yh·η·sk·pm·an·...) 는 ao_step0_ga·ao_floor_n 등이 모든 노드에 새로 씀.
#   예외 = 표본점 버퍼 AF_BX (s.bx): R2·A2 (e5_sh, rt_bar_q mode 0) 가 걸음 시작에 제자리로 옮김 -> 's' 앞에서 사본 (그래프 안 복사), 다시 할 때 되돌림.
#   프레임 시작에 0 으로 채우는 진단 셈 (cnt·qb·awst) 과 제어 (CT, AF_SKIP = rt_ctl op 0 과 같게) 도 되돌림. P2G 결과는 첫 시도 것 그대로 (원자 합 순서 한 판).
#   켠 채 K 1 도 무너지면 (depth > 0) 그 시도 꼬리도 X1 로 건너뛰고 K 2 부터는 split_step 과 같은 쪼개기 (입자 되돌림·프레임 처음부터).
# X3e4 (--s4e T, 결과 바뀜 -- 절 CM39.5 X3e 정의를 W̄·R2·A2 위로): X2 위에서 첫 시도 반복 0 시작 잔차 (ctl op 3 뒤, 'r' 경계) ‖g‖²/F² 가 T 를 넘거나 비유한이면
#   바로 켜고 (AF_WRXON 1) 걸음 시작 ('s') 부터 다시 (X2 와 같은 되돌림 + 예산 시계 TB_RUN 0 -- 앞 반복 0 시간이 C_it 에 안 섞이게). 놓친 무너짐은 X2 경로.
#   X3e 로 켠 시도가 끝에서 무너지면 같은 켬 K 1 을 되풀이하지 않고 K 2 쪼개기로.
class X1:
    on = False
    level = 0
    T = 0.0
    hits = 0
    x2 = 0
    x3e = 0


def _e3sk(s, v):
    import cupy as cp
    try:
        g = s.mod.get_global("X4_E3SK")
    except Exception:
        return
    cp.ndarray((1,), dtype=cp.int32, memptr=g)[0] = int(v)


def _restore(s):
    import cupy as cp
    torch.cuda.synchronize()
    _e3sk(s, 1)
    s.bx.copy_(s._x2bx)
    s.cnt.zero_()
    if getattr(s, "aq", False):
        s.qb.zero_()
    if getattr(s, "lsinv", False) or getattr(s, "y0z", False):
        s.awst.zero_()
    s.CT.zero_()
    cp.ndarray((1,), dtype=cp.int32, memptr=s.mod.get_global("AF_SKIP"))[0] = 0
    torch.cuda.synchronize()


X2C_F = ("U", "vk", "gaA", "mn", "mv", "mqn", "Fq", "Vq", "mq", "xq0", "Sq", "bx")
X2C_I = ("cidx", "cb", "act")


def _x2c_snap(s, first):
    """X2 확인 판 (--s4x 3): 걸음 시작 입력 (P2G 단계 산물) 사본 -- 첫 시도는 uf_end 뒤 ('x', 표본점 버퍼는 's' 앞 사본), 다시 한 시도는 걸음 시작 ('s') 앞."""
    torch.cuda.synchronize()
    IP = s.IP.cpu().tolist()
    nn, nq, nc = int(IP[9]), int(IP[12]), int(IP[11])
    capN, capQ = int(s.capN), int(s.capQc) * 8
    o = dict(IP=IP[:16])
    for nm in X2C_F + X2C_I:
        t = getattr(s, nm, None)
        if not torch.is_tensor(t):
            continue
        if nm == "bx" and first:
            t = s._x2bx
        t = t.reshape(-1)
        n_ = t.numel()
        if nm == "act":
            m = int(IP[14])
        elif nm == "cidx":
            m = int(IP[10])
        elif nm == "cb":
            m = nc * (n_ // max(int(s.capQc), 1))
        elif n_ % capN == 0 and n_ // capN <= 9:
            m = nn * (n_ // capN)
        elif n_ % capQ == 0:
            m = nq * (n_ // capQ)
        else:
            m = n_
        o[nm] = t[:m].clone()
    return o


def _x2c_cmp(a, b):
    r = dict(IP=[int(x != y) for x, y in zip(a["IP"], b["IP"])].count(1))
    for nm in X2C_F + X2C_I:
        if nm not in a or nm not in b:
            continue
        x, y = a[nm], b[nm]
        if x.numel() != y.numel():
            r[nm] = ["크기", x.numel(), y.numel()]
            continue
        if not x.is_floating_point():
            r[nm] = int((x != y).sum())
            continue
        d = (x.double() - y.double()).abs()
        ok = torch.isfinite(d)
        sc = float(x.double().abs()[torch.isfinite(x.double())].max()) if x.numel() else 0.0
        mx = float(d[ok].max()) if bool(ok.any()) else 0.0
        r[nm] = [mx, mx / sc if sc > 0 else 0.0, int((d[ok] > 1e-4 * max(sc, 1e-30)).sum()), int((~ok).sum())]
    return r


class X2C:
    rec = []


def x1_install(level=1, x3e=0.0):
    from . import fem_ao_rt as AORT
    from . import fem_common3 as C3
    from . import fem_wmrx as WRX
    X1.on, X1.level, X1.T = True, int(level), float(x3e)
    assert X1.T <= 0 or X1.level == 2, "--s4e 는 --s4x 2 위"
    cls = AORT.AORTSolver
    o_rec, o_k, o_ctl = cls.record, cls.k, cls.ctl

    def record(s, t_frame, K=None):
        s._x1K = int(K if K is not None else getattr(s, "Kcur", getattr(s, "K", 1)))
        s._x1ks = -1
        cap = getattr(s, "_c3cap", None) is not None
        if cap:
            s._k4b = ["start"]
        elif X1.level >= 2 and getattr(s, "wbar", None) is not None:                 # 데움 (캡처 밖): 사본 버퍼를 지금 용량으로
            if getattr(s, "_x2bx", None) is None or s._x2bx.numel() != s.bx.numel():
                s._x2bx = torch.empty_like(s.bx)
        return o_rec(s, t_frame, K)

    def k(s, name, n, args, b=128):
        if X1.level >= 2 and name == "ao_step0_ga":
            s._x1ks = int(getattr(s, "_x1ks", -1)) + 1
            if s._x1ks == 0 and getattr(s, "_x1K", 0) == 1 and getattr(s, "wbar", None) is not None:
                s._x2bx.copy_(s.bx)                                                    # 걸음 시작 앞 표본점 버퍼 사본 (그래프 안)
                if getattr(s, "_c3cap", None) is not None:
                    C3.mark(s, "s")
        r = o_k(s, name, n, args, b)
        if name == "uf_end" and getattr(s, "_c3cap", None) is not None and getattr(s, "_x1K", 0) == 1:
            C3.mark(s, "x")
        return r

    def ctl(s, op, it=0):
        r = o_ctl(s, op, it)
        if X1.T > 0 and op == 3 and it == 0 and getattr(s, "_c3cap", None) is not None and getattr(s, "_x1K", 0) == 1:
            C3.mark(s, "r")
        return r
    cls.record, cls.k, cls.ctl = record, k, ctl
    o_mark = C3.mark

    def mark(s, kind):
        if getattr(s, "_c3cap", None) is not None and getattr(s, "_k4b", None) is not None:
            s._k4b.append(kind)
        return o_mark(s, kind)
    C3.mark = mark
    o_init, o_rep = C3.SegGraph.__init__, C3.SegGraph.replay

    def __init__(self, s, segs):
        o_init(self, s, segs)
        kb = getattr(s, "_k4b", None)
        self.k4 = list(kb) if (kb is not None and len(kb) == len(segs)) else None
        assert self.k4 is None or (self.k4.count("x") <= 1 and self.k4.count("s") <= 1 and self.k4.count("r") <= 1)
        s._k4b = None

    def replay(self):
        s = self.s
        st = getattr(s, "_s4st", None)
        if self.k4 is None or not ("x" in self.k4 and (getattr(s, "_x1ok", False) or st is not None)):
            return o_rep(self)
        mode = int(s.xg)
        cst = s.c3st
        skip = False
        i = 0
        si = self.k4.index("s") if "s" in self.k4 else None
        if st is not None:                                                             # 재생 시작 (step 의 e0 바로 뒤) -- 첫 시도 몫 시간 나누기용
            st["e0"] = torch.cuda.Event(enable_timing=True); st["e0"].record()
        while i < len(self.segs):
            cond, g = self.segs[i]
            kd = self.k4[i]
            if kd == "r" and st is not None and X1.T > 0 and not st["jump"] and st["won"] == 0 and si is not None and int(s.IP[13]) == 0:
                g2, F2 = float(s.SC[1]), float(s.CD[2])
                if not (g2 <= X1.T * F2):                                              # X3e4: 반복 0 시작 잔차로 무너짐 판정 -> 켜고 걸음 시작부터
                    ev = torch.cuda.Event(enable_timing=True); ev.record(); st["ev"] = ev
                    _restore(s)
                    from . import fem_speed3 as S3
                    if S3.TB.mod is not None:
                        S3.TB.g("TB_RUN", np.int32)[0] = 0
                        torch.cuda.synchronize()
                    WRX.on(s, 1); st["won"] = 1; st["jump"] = "x3e"; X1.x3e += 1
                    i, skip = si, False
                    continue
            if kd == "s" and X1.level == 3 and int(s.IP[13]) == 0:
                if getattr(s, "_x1won", 0) == 0:                                       # X2 확인 (가): 첫 시도 걸음 시작 앞 값
                    s._x2s = _x2c_snap(s, False)
                elif getattr(s, "_x2c", None) is not None:                              # X2 확인 (나): 다시 한 시도 (P2G 새로) 걸음 시작 앞 값 대 첫 시도 'x' 시점 값
                    X2C.rec.append(dict(kind="다시", **_x2c_cmp(s._x2c, _x2c_snap(s, False))))
                    s._x2c = None
            if kd == "x" and int(s.IP[13]) == 0:
                coll = C3.nfail(s, s.ufl, s.ufg, 1) > 0
                if X1.level == 3 and getattr(s, "_x1won", 1) == 0 and getattr(s, "_x2s", None) is not None:
                    xs = _x2c_snap(s, False)                                            # (가) 같은 시도 's' 대 'x' (bx 는 걸음 안에서 옮겨짐 -> 사본 대 's' 값을 따로)
                    r_ = _x2c_cmp(s._x2s, xs)
                    r_["bx_snap"] = _x2c_cmp(dict(IP=s._x2s["IP"], bx=s._x2s["bx"]), dict(IP=s._x2s["IP"], bx=s._x2bx.reshape(-1)[:s._x2s["bx"].numel()].clone())).get("bx")
                    X2C.rec.append(dict(kind="같은 시도", coll=bool(coll), **r_))
                    s._x2s = None
                    if coll:
                        s._x2c = _x2c_snap(s, True)
                if coll and st is not None and not st["jump"] and st["won"] == 0 and si is not None:
                    ev = torch.cuda.Event(enable_timing=True); ev.record(); st["ev"] = ev   # X2: 첫 시도 무너짐 -> 켜고 걸음 시작부터 (같은 재생 안)
                    st["nf0"] = 1
                    _restore(s)
                    WRX.on(s, 1); st["won"] = 1; st["jump"] = "x2"; X1.x2 += 1
                    i, skip = si, False
                    continue
                discard = getattr(s, "_x1ok", False) if st is None else (coll and (st["depth"] > 0 or st["won"] == 0))
                if coll and discard:
                    X1.hits += 1
                    return
            if not cond:                                                                 # 아래는 fem_common3.SegGraph.replay 글자 그대로
                skip = False
                if mode == 2:
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record(); g.replay(); e1.record(); cst["ev"].append((0, e0, e1))
                else:
                    g.replay()
                i += 1
                continue
            if not skip:
                skip = bool(int(s.CT[0]) != 0)
            cst["sk" if skip else "run"] += 1
            if mode == 1 and skip:
                i += 1
                continue
            if mode == 2:
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); g.replay(); e1.record(); cst["ev"].append((2 if skip else 1, e0, e1))
            else:
                g.replay()
            i += 1
    C3.SegGraph.__init__, C3.SegGraph.replay = __init__, replay
    o_on = WRX.on

    def on(s, v):
        s._x1won = int(v)
        return o_on(s, v)
    WRX.on = on
    o_split = AORT.split_step

    def split_step(s, t, h, sub_dt, depth):
        col = bool(getattr(s, "wrx", None)) and getattr(s, "_x1won", None) == 0      # X3c 첫 시도 (끈 채): 무너지면 되돌려 켠 채 다시
        prev = getattr(s, "_x1ok", False)
        s._x1ok = (int(depth) > 0) or col
        try:
            return o_split(s, t, h, sub_dt, depth)
        finally:
            s._x1ok = prev
    AORT.split_step = split_step
    if X1.level != 2:
        return
    o_init = cls.__init__

    def __init__(s, *a, **kw):                                                         # X2 다시 재생 구간의 W̄ 모으기 (e3_*, 걸음 시작 앞 산물만 읽음 -> 같은 값) 건너뛰기 표시
        if not getattr(AORT, "_x2_src", False):
            AORT._x2_src = True
            o_s = AORT._src

            def _src(*a_, **k_):
                r_ = o_s(*a_, **k_)
                if "e3_wq(" not in r_:
                    return r_
                r_ = r_.replace('extern "C" {', 'extern "C" {\n__device__ int X4_E3SK;\n', 1)
                for nm in ("e3_wqz", "e3_wq", "e3_wqn"):
                    r_ = _ins_body(r_, nm, " if (X4_E3SK) return;")
                return r_
            AORT._src = _src
        return o_init(s, *a, **kw)
    cls.__init__ = __init__
    o_col = WRX.split_col

    def split_col(s, t, h, sub_dt, depth, fn=None):
        """WRX.split_col 과 같은 돌려줌 (다시 했는지, K, 시도 [(K, 무너진 걸음, ms, 켬)]) -- 켠 채 K 1 은 같은 재생 안 (X2·X3e4), K 2 부터는 split_step 과 같음."""
        if fn is not None or not isinstance(s, cls):
            return o_col(s, t, h, sub_dt, depth, fn)
        import math
        bk = s._bk()
        WRX.on(s, 0)
        base = (s.ga_c, s.nt, s.cg)
        AORT.set_K_ga(s, 1, h, sub_dt)
        st = s._s4st = dict(jump=None, won=0, depth=int(depth), ev=None, nf0=None, e0=None)
        _e3sk(s, 0)
        torch.cuda.synchronize()
        try:
            redo = s.step(t)
        finally:
            s._s4st = None
            torch.cuda.synchronize()
            _e3sk(s, 0)
            torch.cuda.synchronize()
        ms = float(s.last_ms)
        nf = C3.nfail(s, s.ufl, s.ufg, 1)
        if st["jump"] is None:
            tries = [(1, nf, ms, 0)]
            if nf == 0 or not math.isfinite(ms):
                return redo, 1, tries
            s._rs(bk)                                                                  # (넘침으로 같은 재생 안 다시 하기를 못 한 경우) 원래 X3c 경로
            WRX.on(s, 1)
            r1, K1, tr1 = AORT.split_step(s, t, h, sub_dt, depth)
            WRX.on(s, 0)
            s.last_ms = ms + float(s.last_ms)
            return (redo or r1), K1, tries + [(a, b, c, 1) for (a, b, c) in tr1]
        torch.cuda.synchronize()
        m1 = st["e0"].elapsed_time(st["ev"]) if (st["ev"] is not None and st["e0"] is not None) else 0.0
        m1 = min(max(m1, 0.0), ms)
        tries = [(1, st["nf0"] if st["jump"] == "x2" else -1, m1, 0), (1, nf, ms - m1, 1)]
        tot = ms
        K = 1
        if nf > 0 and math.isfinite(ms) and depth > 0:                               # 켠 채 K 1 도 무너짐 -> K 2 부터 (split_step 과 같은 글자, 켠 채)
            wf = getattr(s, "wf", {})
            for dp in range(1, depth + 1):
                K = 1 << dp
                s._rs(bk)
                if wf.get("ks_ga") is not None:
                    s.ga_c = wf["ks_ga"]
                if wf.get("ks_nt"):
                    s.nt, s.cg = int(wf["ks_nt"]), int(wf["ks_cg"])
                AORT.set_K_ga(s, K, h, sub_dt)
                redo = s.step(t) or redo
                m_ = float(s.last_ms)
                nf_ = C3.nfail(s, s.ufl, s.ufg, K)
                tries.append((K, nf_, m_, 1))
                tot += m_
                if nf_ == 0 or not math.isfinite(m_):
                    break
            s.ga_c, s.nt, s.cg = base
        WRX.on(s, 0)
        s.last_ms = tot
        return redo, K, tries
    WRX.split_col = split_col


# ======================= (S2) 걸음 시작 노드별 β (y0loc, yl_*) 확인 라운드 조기 끝 (결과 불변) =======================
# 순서 (fem_ao_rt.record, --y0loc R): yl_kq -> [yl_mix -> yl_chk(r) -> yl_merge] x R -> yl_mix -> yl_chk(R) -> yl_fb -> yl_mix -> yl_st.
# yl_chk 가 어긴 표본점 0 (fl = 0) 이면 kn2 에 아무것도 안 흩음. kn2 는 앞 라운드 값만 들고 있고 그 값은 이미 yl_merge 로 kn 에 들어가 있어 (kn ≥ kn2)
#   뒤 yl_merge·yl_mix (같은 kn·eU·eS -> 같은 y·η)·yl_chk (같은 y -> 0)·yl_fb (fl 0 -> 아무것도 안 함) 는 모두 같은 값 -> 건너뛰어도 kn·y·η·yfl 비트 같음.
# 장치 표시 YL4_DN (yl_kq 가 0 으로, yl_chk 뒤 yl4_chk 가 fl 0 이면 1), yl_mix·yl_chk·yl_merge·yl_fb 첫머리 'YL4_DN 이면 돌아감'. --s4y 2 = 확인 판 (건너뛰지 않고 YL4_DN 뒤 어긴 수 셈).
YL4_HEAD = "__device__ int YL4_DN; __device__ unsigned long long YL4_BAD, YL4_NCLEAN, YL4_NCHK;\n"
YL4_K = r"""
__global__ void yl4_chk(const int* fl) {
    if (AF_SKIP) return;
    YL4_NCHK++;
    if (!YL4_DN && fl[0] == 0) { YL4_DN = 1; YL4_NCLEAN++; }
}
"""


class YL4:
    mode = 0


def yl4_src(src):
    if "yl_kq(" not in src:
        return src
    src = src.replace('extern "C" {', 'extern "C" {\n' + YL4_HEAD, 1)
    src = _ins_body(src, "yl_kq", "\n    if (blockIdx.x == 0 && threadIdx.x == 0) YL4_DN = 0;                               // 절 CM50 S2\n")
    if YL4.mode == 1:
        for nm in ("yl_mix", "yl_chk", "yl_merge", "yl_fb"):
            src = _ins_body(src, nm, " if (YL4_DN) return;")
    else:
        m = re.search(r"__global__\s+void\s+(?:__launch_bounds__\(\d+\)\s+)?yl_chk\s*\(", src)
        j = src.index("\n}\n", m.end())
        body = src[m.start():j]
        assert body.count("atomicAdd(fl, 1);") == 1
        src = src[:m.start()] + body.replace("atomicAdd(fl, 1);", "atomicAdd(fl, 1); if (YL4_DN) atomicAdd(&YL4_BAD, 1ull);", 1) + src[j:]
    i = src.rfind("}")
    return src[:i] + YL4_K + src[i:]


def yl4_install(mode):
    from . import fem_ao_rt as AORT
    YL4.mode = int(mode)
    cls = AORT.AORTSolver
    o_init, o_k = cls.__init__, cls.k

    def __init__(s, *a, **kw):
        if not getattr(AORT, "_yl4_src", False):
            AORT._yl4_src = True
            o_s = AORT._src
            AORT._src = lambda *a_, **k_: yl4_src(o_s(*a_, **k_))
        return o_init(s, *a, **kw)

    def k(s, name, n, args, b=128):
        r = o_k(s, name, n, args, b)
        if name == "yl_chk":                                                           # 마지막 인자 = 이 라운드 fl (P(s.yfl[r:]))
            o_k(s, "yl4_chk", 1, (args[-1],), 1)
        return r
    cls.__init__, cls.k = __init__, k


def yl4_counts(s):
    import cupy as cp
    g = lambda nm: cp.ndarray((1,), dtype=cp.uint64, memptr=s.mod.get_global(nm))
    o = {}
    for nm, key in (("YL4_NCHK", "chk"), ("YL4_NCLEAN", "clean"), ("YL4_BAD", "bad")):
        a = g(nm)
        o[key] = int(a[0])
        a[0] = 0
    return o


# ======================= 프레임 기록 =======================
def frame_install(out):
    """쪼개기 함수 (프레임 진입) 를 감싸 프레임마다 s4.jsonl 한 줄 (단계 시간 · S1 셈)."""
    from . import fem_ao_rt as AORT
    from . import fem_wmrx as WRX
    log = open(os.path.join(out, "s4.jsonl"), "w")
    dep, fr = [0], [0]

    def outer(fn):
        def w(s, *a, **kw):
            if dep[0] == 0 and SP.on:
                torch.cuda.synchronize()
                sp_g("SP_N", np.int32)[0] = 0
                torch.cuda.synchronize()
            dep[0] += 1
            try:
                r = fn(s, *a, **kw)
            finally:
                dep[0] -= 1
            if dep[0] == 0 and isinstance(s, AORT.AORTSolver):
                rec = dict(t=fr[0])
                fr[0] += 1
                if SP.on:
                    rec["sp"] = sp_frame()
                    if SP.unk:
                        rec["unk"] = sorted(n_ for _, n_ in SP.unk)
                        SP.unk.clear()
                if WS4.mode:
                    rec["ws4"] = ws4_counts(s)
                if YL4.mode:
                    rec["yl4"] = yl4_counts(s)
                if X1.on:
                    rec["x1"] = [X1.hits, X1.x2, X1.x3e]
                    X1.hits = X1.x2 = X1.x3e = 0
                    if X2C.rec:
                        rec["x2c"] = X2C.rec
                        X2C.rec = []
                log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()
            return r
        return w
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)
