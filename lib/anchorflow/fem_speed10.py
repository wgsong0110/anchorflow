"""절 CM78 묶음 speed10 (11-08 위임, 수박 렌더 포함 벽시계 60 fps 이어서): 공통 후보 H5 (= H4 + C + Y, 절 CM75·CM76) 위
(N) 노드 상자 압축 (결과 불변 -- 값 대조) 과 (K) 넘침 다시 할당·다시 캡처 프레임 (~110 ms 호스트) 없애기. 감싸개 exe/speed10_wrap.py 가 부름.

(N) 노드 상자 압축 (--s10n 1, 확인 판 --s10n 2; 옛 실행기 = 수박만): 노드 커널 가운데 머리가 'int n = blockIdx.x*blockDim.x + threadIdx.x; if (n >= IP[9]) return;'
    이고 본문에 블록 협동 (__syncthreads·__shared__·blocksum·워프 *_sync)·원자 연산·스레드 번호 (blockIdx·threadIdx·gridDim)·'n == 0' 이 없는 것을 골라
    (--s10k 이름,.. 또는 all) 본문을 장치 함수 NAME__s10b(인자.., n) 로 옮기고 커널은 격자 보폭 고리로 그 함수를 부름:
      S10N == 1 && S10RDY (질량 있는 노드 목록 AO_ACT[0..IP[14]) 이 지금 상자로 만들어짐) 이면 목록 노드만, 아니면 상자 전체 n < IP[9] (= 원래와 같은 일).
    S10RDY 는 ao_act 를 띄운 바로 뒤 1, rt_reset·rt_box·ao_act_reset 앞 0 (호스트 감싸개가 같은 스트림에 1 스레드 커널). 고친 커널은 격자를 --s10g 블록으로
    줄여 띄움 (격자 보폭 고리라 상자·용량과 무관하게 맞음; 빈 블록 띄우기 몫을 없앰).
    확인 판 (--s10n 2): 고친 커널을 부를 때마다 그 커널의 모든 포인터 인자가 가리키는 버퍼 (풀이기 텐서의 저장소 전체) 를 그림자에 떠 두고
      (가) S10N = 1 로 압축 판을 진짜 버퍼에, (나) S10N = 0 으로 같은 커널 (상자 전체) 을 그림자 버퍼에 돌린 뒤 버퍼마다 진짜 대 그림자를 32 비트 낱말로 비트 대조
      -> 커널마다 '대조한 부름 수 · 다른 낱말 수' (s10.jsonl). 모든 부름에서 다름 0 이면 같은 앞 상태에서 압축 판 = 원래 판 (빈 노드에서 원래 커널이 쓰는 값이
      이미 그 자리에 있음) 이고, 고친 커널 전부가 그렇다면 같은 실행의 궤적이 원래 계산과 비트 단위로 같음 (귀납). 포인터를 풀이기 텐서에 못 맞춘 부름은 '못 맞춤' 으로 셈.
    목록 고르기: 확인 판을 후보 전부 (all) 로 돌려 다름이 나온 커널을 빼고, 남은 목록으로 확인 판을 다시 돌려 다름 0 을 확인 (exe/speed10_klist.txt).
(K) 첫 할당 상자 배수 (--s10cap M): 옛 실행기 (AORTSolver) alloc(None) 때 노드 상자·칸 상자 용량을 지금 상자 x M 으로 (표본점 용량 = 점유 칸 x cap_mul 그대로).
    넘침 (IP[13]) 이면 프레임을 되돌려 다시 할당·다시 캡처하는 길 (절 AN step: 그래프 재생 5 번 더 + 캡처 ~100 ms 호스트) 을 덜 타게 함. 인덱스는 상자로만
    정해지고 용량은 버퍼 크기·격자 크기만 바꿈 (넘침 없는 동안 같은 계산; 절 CM72 새 실행기 배수 2·4 결정적 해시 같음).
"""
import bisect
import json
import os
import re

import numpy as np
import torch


class C:
    out = "."
    n = 0
    k = "all"
    cap = 0.0
    g = 1024
    tx = []
    skip = {}
    said = False
    log = None
    solver = None
    unm = {}
    shadow = {}
    bmap = None
    nchk = 0


HDR = r"""
// ---- 절 CM78 speed10 ----
__device__ int S10N = 0;
__device__ int S10RDY = 0;
__device__ unsigned long long* S10CNT;
"""

TAIL = r"""
// ---- 절 CM78 speed10 커널 ----
__global__ void s10_set(int v) { S10N = v; }
__global__ void s10_rdy(int v) { S10RDY = v; }
__global__ void s10_tick(int slot) { S10CNT[slot*2] += 1ULL; }
__global__ void s10_cpy(unsigned* d, const unsigned* s, long long nw) {
    long long st = (long long)gridDim.x*blockDim.x;
    for (long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x; i < nw; i += st) d[i] = s[i];
}
__global__ void s10_cmp(const unsigned* a, const unsigned* b, long long nw, int slot) {
    long long st = (long long)gridDim.x*blockDim.x; unsigned long long c = 0;
    for (long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x; i < nw; i += st) c += (a[i] != b[i]) ? 1ULL : 0ULL;
    if (c) atomicAdd(&S10CNT[slot*2+1], c);
}
"""

H1 = re.compile(r"int n = blockIdx\.x\s*\*\s*blockDim\.x \+ threadIdx\.x;\s*if \(n >= IP\[9\]\) return;")
BAD = ("__syncthreads", "__shared__", "blocksum", "_sync(", "atomic", "blockIdx", "threadIdx", "gridDim")
HEAD = re.compile(r"__global__\s+void\s+(__launch_bounds__\(\d+\)\s+)?(\w+)\s*\(")


def _match(src, i, o, c):
    """src[i] == o 인 자리에서 짝 c 의 자리 (주석·글자 건너뜀)."""
    d, j, n = 0, i, len(src)
    while j < n:
        ch = src[j]
        if src.startswith("//", j):
            j = src.find("\n", j)
            j = n if j < 0 else j
            continue
        if src.startswith("/*", j):
            j = src.find("*/", j) + 2
            continue
        if ch in "\"'":
            q = ch
            j += 1
            while j < n and src[j] != q:
                j += 2 if src[j] == "\\" else 1
            j += 1
            continue
        if ch == o:
            d += 1
        elif ch == c:
            d -= 1
            if d == 0:
                return j
        j += 1
    return -1


def _pnames(params):
    out = []
    for p in params.split(","):
        p = re.sub(r"/\*.*?\*/", "", p).strip()
        m = re.search(r"(\w+)\s*$", p)
        if not m:
            return None
        out.append(m.group(1))
    return out


def transform(src, want):
    """고를 수 있는 노드 커널을 장치 함수 + 격자 보폭 고리 커널로. 돌려주는 것: (새 소스, 고친 이름 목록, 뺀 이름 -> 까닭)."""
    tx, skip = [], {}
    pos, last = 0, 0
    out = []
    while True:
        m = HEAD.search(src, pos)
        if not m:
            break
        nm = m.group(2)
        p0 = m.end() - 1
        p1 = _match(src, p0, "(", ")")
        b0 = src.find("{", p1)
        if p1 < 0 or b0 < 0 or src[p1 + 1:b0].strip():
            pos = m.end()
            continue
        b1 = _match(src, b0, "{", "}")
        if b1 < 0:
            pos = m.end()
            continue
        inner = src[b0 + 1:b1]
        h = H1.search(inner)
        if h is None or (want is not None and nm not in want):
            pos = b1 + 1
            continue
        pre, rest = inner[:h.start()], inner[h.end():]
        why = [t for t in BAD if t in pre or t in rest]
        pre_ = re.sub(r"/\*.*?\*/", "", re.sub(r"//[^\n]*", "", pre), flags=re.S)
        if re.search(r"\bn\b", pre_):
            why.append("n-앞")
        if "{" in pre_ or re.search(r"\b(for|while|do)\b", pre_):
            why.append("앞-블록")
        if re.search(r"\bn\s*==\s*0\b", rest):
            why.append("n==0")
        params = src[p0 + 1:p1]
        names = _pnames(params)
        if names is None or "IP" not in names:
            why.append("인자")
        if why:
            skip[nm] = why
            pos = b1 + 1
            continue
        lb = m.group(1) or ""
        a = ", ".join(names)
        dev = f"static __device__ __forceinline__ void {nm}__s10b({params}, int n) {{{pre}{rest}}}\n"
        ker = (f"__global__ void {lb}{nm}({params}) {{   // 절 CM78 (N): 원래 본문 = {nm}__s10b\n"
               f"    int s10_st = gridDim.x*blockDim.x, s10_g = blockIdx.x*blockDim.x + threadIdx.x;\n"
               f"    if (S10N == 1 && S10RDY) {{ int s10_na = IP[14]; for (int s10_i = s10_g; s10_i < s10_na; s10_i += s10_st) {nm}__s10b({a}, AO_ACT[s10_i]); }}\n"
               f"    else {{ int s10_nb = IP[9]; for (int s10_n = s10_g; s10_n < s10_nb; s10_n += s10_st) {nm}__s10b({a}, s10_n); }}\n"
               f"}}")
        out.append(src[last:m.start()])                                                   # '__global__' 앞 글자는 그대로, 커널 자리만 바꿈
        out.append(dev + ker)
        tx.append(nm)
        pos = last = b1 + 1
    out.append(src[last:])
    return "".join(out), tx, skip


def _src(code):
    if not isinstance(code, str) or "s10_set(" in code or "__device__ int* AO_ACT;" not in code or "ao_act(" not in code:
        return code
    want = None if C.k == "all" else set(q for q in C.k.split(",") if q)
    code, tx, skip = transform(code, want)
    k = code.find("__global__")
    k = code.rfind("\n", 0, k) + 1
    code = code[:k] + HDR + code[k:]
    e = code.rstrip().rfind("\n}")
    code = code[:e] + "\n" + TAIL + code[e:]
    C.tx, C.skip = tx, skip
    if not C.said:
        C.said = True
        print(f"[절 CM78] 노드 압축 고친 커널 {len(tx)} 개 {tx}; 뺀 후보 {len(skip)} 개 {skip}", flush=True)
    return code


class FnProxy:
    def __init__(s_, mp, name, f):
        s_.mp, s_.name, s_.f = mp, name, f

    def __getattr__(s_, a):
        return getattr(s_.f, a)

    def __call__(s_, grid, block, args, **kw):
        nm = s_.name
        mp = s_.mp
        if C.n and nm in ("rt_reset", "rt_box", "ao_act_reset"):
            mp.raw("s10_rdy")((1,), (1,), (np.int32(0),))
        if C.n and nm in C.tx:
            g2 = (min(int(grid[0]), int(C.g)),) + tuple(grid[1:])
            if C.n == 2:
                return _check(mp, s_, g2, block, args, kw)
            return s_.f(g2, block, args, **kw)
        r = s_.f(grid, block, args, **kw)
        if C.n and nm == "ao_act":
            mp.raw("s10_rdy")((1,), (1,), (np.int32(1),))
        return r


class ModProxy:
    def __init__(s_, m):
        s_._m = m
        s_._f = {}

    def raw(s_, name):
        return s_._m.get_function(name)

    def get_function(s_, name):
        f = s_._f.get(name)
        if f is None:
            f = FnProxy(s_, name, s_._m.get_function(name))
            s_._f[name] = f
        return f

    def __getattr__(s_, a):
        return getattr(s_._m, a)


def _bufs(s):
    """풀이기에서 닿는 장치 버퍼 (저장소 시작, 바이트) 목록."""
    seen = {}

    def add(t):
        try:
            if torch.is_tensor(t) and t.is_cuda:
                st = t.untyped_storage()
                seen[int(st.data_ptr())] = max(seen.get(int(st.data_ptr()), 0), int(st.nbytes()))
        except Exception:
            pass

    def walk(v, d):
        if torch.is_tensor(v):
            add(v)
        elif d < 2 and isinstance(v, (list, tuple)):
            for q in v:
                walk(q, d + 1)
        elif d < 2 and isinstance(v, dict):
            for q in v.values():
                walk(q, d + 1)
    for v in list(vars(s).values()):
        walk(v, 0)
    b = sorted((p, n) for p, n in seen.items() if p)
    return [q[0] for q in b], b


def _find(ptr):
    if C.bmap is None:
        return None
    ks, b = C.bmap
    i = bisect.bisect_right(ks, ptr) - 1
    if i >= 0 and b[i][0] <= ptr < b[i][0] + b[i][1]:
        return b[i]
    return None


def _shadow(base, nb, dev):
    sh = C.shadow.get(base)
    if sh is None or sh.numel() * 4 < nb:
        sh = torch.empty((nb + 3) // 4, device=dev, dtype=torch.int32)
        C.shadow[base] = sh
    return sh


def _check(mp, fp, grid, block, args, kw):
    s = C.solver
    slot = C.tx.index(fp.name)
    if s is None:
        C.unm[fp.name] = C.unm.get(fp.name, 0) + 1
        return fp.f(grid, block, args, **kw)
    ptrs = []
    for i, a in enumerate(args):
        if isinstance(a, np.uint64) and int(a) != 0:
            ptrs.append((i, int(a)))
    hit = {}
    for _, p in ptrs:
        b = _find(p)
        if b is None:
            C.bmap = _bufs(s)
            b = _find(p)
        if b is None:
            C.unm[fp.name] = C.unm.get(fp.name, 0) + 1
            mp.raw("s10_set")((1,), (1,), (np.int32(1),))
            r = fp.f(grid, block, args, **kw)
            mp.raw("s10_set")((1,), (1,), (np.int32(0),))
            return r
        hit[b[0]] = b
    dev = s.x.device
    GB = (2048,)
    for base, (b0, nb) in hit.items():
        sh = _shadow(base, nb, dev)
        mp.raw("s10_cpy")(GB, (256,), (np.uint64(int(sh.data_ptr())), np.uint64(base), np.int64(nb // 4)))
    a2 = list(args)
    for i, p in ptrs:
        b = _find(p)
        a2[i] = np.uint64(int(C.shadow[b[0]].data_ptr()) + (p - b[0]))
    mp.raw("s10_set")((1,), (1,), (np.int32(1),))
    r = fp.f(grid, block, args, **kw)
    mp.raw("s10_set")((1,), (1,), (np.int32(0),))
    fp.f(grid, block, tuple(a2), **kw)
    mp.raw("s10_tick")((1,), (1,), (np.int32(slot),))
    for base, (b0, nb) in hit.items():
        sh = C.shadow[base]
        mp.raw("s10_cmp")(GB, (256,), (np.uint64(base), np.uint64(int(sh.data_ptr())), np.int64(nb // 4), np.int32(slot)))
    C.nchk += 1
    return r


def pre_install(n=0, k="all", g=1024):
    """감싸개가 뜨는 순간 (다른 감싸개의 소스 고침보다 먼저 = 가장 안쪽) RawModule 을 감쌈 -> 마지막 소스를 보고 고침, 주 모듈은 ModProxy 로."""
    import cupy as cp
    C.n, C.k, C.g = int(n), str(k), int(g)
    if not C.n:
        return
    o_rm = cp.RawModule

    def RawModule(*a, **kw):
        code = kw.get("code", a[0] if a else None)
        main = isinstance(code, str) and "__device__ int* AO_ACT;" in code and "ao_act(" in code and "s10_set(" not in code
        if main:
            code = _src(code)
            if "code" in kw:
                kw["code"] = code
            else:
                a = (code,) + tuple(a[1:])
        m = o_rm(*a, **kw)
        return ModProxy(m) if main else m
    cp.RawModule = RawModule


def bind(s):
    import cupy as cp
    m = s.mod
    try:
        m.get_global("S10N")
    except Exception:
        return
    if getattr(s, "s10c", None) is None:
        s.s10c = torch.zeros(2 * max(1, len(C.tx)) + 2, device=s.x.device, dtype=torch.int64)
    cp.ndarray((1,), dtype=cp.uint64, memptr=m.get_global("S10CNT"))[0] = int(s.s10c.data_ptr())
    cp.ndarray((1,), dtype=cp.int32, memptr=m.get_global("S10N"))[0] = 1 if C.n == 1 else 0
    cp.ndarray((1,), dtype=cp.int32, memptr=m.get_global("S10RDY"))[0] = 0
    C.solver = s
    C.bmap = _bufs(s)


def cap_install(M):
    """(K) 첫 할당 상자 배수 + 다시 할당 기록 (alloc(None) 부를 때마다 앞 IP[:16]·용량, 뒤 용량 -> s10.jsonl 'ovf'; 절 CM72 s7ovf 와 같은 셈)."""
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    oa = ART.RTSolver.alloc

    def alloc(s, need, *a, **kw):
        if need is None and isinstance(s, AORT.AORTSolver):
            rec = dict(ovf=1)
            try:
                if getattr(s, "IP", None) is not None and getattr(s, "capN", None) is not None:
                    rec.update(ip=[int(v) for v in s.IP[:16].tolist()], old=[int(s.capN), int(s.capC), int(s.capQc)], cap_mul=float(s.cap_mul))
            except Exception as e_:
                rec["err"] = str(e_)[:200]
            if M > 0:
                dev = s.x.device
                y = s.x.double() @ s.Ai.double().T
                b = torch.floor(y - 0.5)
                nb = (b.max(0).values - b.min(0).values + 3).long().tolist()
                nn = nb[0] * nb[1] * nb[2]
                ncb = (nb[0] - 2) * (nb[1] - 2) * (nb[2] - 2)
                uc = torch.unique(((b - b.min(0).values).long() * torch.tensor([(nb[1] - 2) * (nb[2] - 2), nb[2] - 2, 1], device=dev)).sum(1)).numel()
                s.nb0 = nb
                cm = float(s.cap_mul)
                r = oa(s, (nn * M / cm, ncb * M / cm, uc), *a, **kw)
                rec.update(need=[nn, ncb, uc], M=M)
                print(f"[절 CM78] 할당 상자 배수 {M}: 상자 노드 {nn} 칸 {ncb} 점유 칸 {uc} (cap_mul {cm}) -> capN {s.capN} capC {s.capC} capQc {s.capQc}", flush=True)
            else:
                r = oa(s, need, *a, **kw)
            rec["new"] = [int(s.capN), int(s.capC), int(s.capQc)]
            if C.log is not None:
                C.log.write(json.dumps(rec) + "\n")
                C.log.flush()
            return r
        return oa(s, need, *a, **kw)
    ART.RTSolver.alloc = alloc


def frame_end(s, fr):
    rec = dict(f=fr)
    if fr <= 3:
        rec.update(tx=list(C.tx), skip=C.skip, nchk=C.nchk)
    if C.unm:
        rec["unm"] = dict(C.unm)
        C.unm = {}
    if C.n == 2 and getattr(s, "s10c", None) is not None:
        torch.cuda.synchronize()
        c = s.s10c.tolist()
        s.s10c.zero_()
        rec["chk"] = {nm: [c[2 * i], c[2 * i + 1]] for i, nm in enumerate(C.tx) if c[2 * i] or c[2 * i + 1]}
    rec["cap"] = [int(getattr(s, "capN", 0)), int(getattr(s, "capC", 0)), int(getattr(s, "capQc", 0))]
    if C.log is not None:
        C.log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        C.log.flush()


def frame_install():
    from . import fem_an_rt as ART
    from . import fem_ao_rt as AORT
    from . import fem_wmrx as WRX
    from . import fem_speed6 as S6
    dep = [0]

    def outer(fn):
        def w(s, *a, **kw):
            dep[0] += 1
            try:
                return fn(s, *a, **kw)
            finally:
                dep[0] -= 1
                if dep[0] == 0 and isinstance(s, AORT.AORTSolver):
                    frame_end(s, S6.W.fr)
        return w
    ART.split_step = outer(ART.split_step)
    AORT.split_step = outer(AORT.split_step)
    WRX.split_col = outer(WRX.split_col)


def install(out, cap=0.0):
    """실행기를 runpy 로 띄우는 순간, 다른 감싸개 설치 뒤 (가장 바깥) 에 부름."""
    from . import fem_ao_rt as AORT
    C.out, C.cap = out, float(cap)
    C.log = open(os.path.join(out, "s10.jsonl"), "w")
    cap_install(C.cap)
    if C.n:
        cls = AORT.AORTSolver
        ob = cls.bind_act

        def bind_act(s, *a, **kw):
            rr = ob(s, *a, **kw)
            try:
                bind(s)
            except Exception as e_:
                C.skip["bind"] = [str(e_)[:200]]
            return rr
        cls.bind_act = bind_act
    frame_install()
    print(f"[절 CM78] speed10 켬 노드 압축 {C.n} 목록 {C.k if len(C.k) < 80 else C.k[:80] + '..'} 격자 {C.g} 첫 할당 상자 배수 {C.cap}", flush=True)
