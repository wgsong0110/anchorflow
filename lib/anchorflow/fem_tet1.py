"""절 CM102 (사용자 11-10 승인 1 안): 공용 실행기 (새 fem_an_rt · 옛 fem_an_rt_v0/fem_ao_rt) 의 기저만 사면체 일차 기저로 바꾸는 판.

정의 (doc/fem_realtime.md 절 CM102.0):
- 격자·노드는 그대로 (SX.lattice, 노드 = 격자 정수점 A k). 칸을 이차 B-스플라인의 '노드 가운데 칸' [b+½, b+1½)³ 대신
  격자 칸 [b, b+1)³ (꼭짓점 = 노드) 으로: 입자 칸 b = floor(y) (원래 floor(y − ½)), 칸 안 좌표 f = y − b ∈ [0,1)³.
  노드 27 칸 표 (b + {0,1,2}³) 는 그대로 두고 꼭짓점 b + {0,1}³ 만 씀 (b + 2 줄은 값 0).
- 칸을 Kuhn 사면체 6 개로 (simplex.py locate 와 같은 규칙: f 를 내림차순 정렬한 축 순서 (p0,p1,p2), 꼭짓점 0 · e_p0 · e_p0+e_p1 · (1,1,1),
  barycentric λ = (1−s0, s0−s1, s1−s2, s2)), 사면체 종류 t = 2 p0 + [p1 > p2] (0..5). 같은 값이면 축 번호 작은 쪽이 앞 (안정 정렬).
- 적분점: 사면체마다 1 개 (무게중심, 격자 좌표 p0 ¾ · p1 ½ · p2 ¼), 팔분 자리 8 개 중 0..5 를 사면체로, 6·7 은 빈 자리 (부피·질량 0).
  부피·질량: 칸 합 / 8 대신 칸 합 / 6 (점유 가중 κ 의 '칸 몫' 도 / 6), 실제 몫은 그 사면체 안 입자 합 (원래 팔분 칸 안 입자 합과 같은 식).
- 기저 표 BW (값) · BG (월드 기울기): 사면체 t 의 네 꼭짓점에 ¼ · ∇λ (상수), 나머지 23 노드 0; 자리 6·7 은 모두 0.
- 입자 P2G·G2P 가중 = 그 입자 사면체의 λ (네 노드), 기울기 = 그 사면체 ∇λ (입자 F 갱신은 사면체 상수 기울기).
- 가우스 점 보간 교체 (--fp_interp 입자 F · --lj_interp 입자 logJp) = 자기 사면체 적분점 값을 그대로 (삼선형 가중 대신 자기 자리 1).
- 거친 MG 층 (전처리기) 은 그대로 (칸 위치 반 칸 어긋남은 전처리기 품질만 바꿈, 풀리는 식은 그대로).

설치: install(mode) -- mode 0 이면 소스를 바꾸지 않고 컴파일되는 소스의 해시만 기록 (끈 판 = H9 와 같은 글자 확인용), 1 이면 바꿈.
"""
import hashlib
import json
import os
import re

import numpy as np

DEV = r'''
#ifndef AF_TET1_DEF
#define AF_TET1_DEF
// 절 CM102 사면체 일차 기저 (Kuhn 6 분할, 칸 [b, b+1)^3, 노드 27 표의 b + {0,1}^3 만)
__device__ __forceinline__ void af_perm(const float* f, int* pr, float* sv) {
    int a = 0, b = 1, c = 2, t;
    if (f[b] > f[a]) { t = a; a = b; b = t; }
    if (f[c] > f[b]) { t = b; b = c; c = t; }
    if (f[b] > f[a]) { t = a; a = b; b = t; }
    pr[0] = a; pr[1] = b; pr[2] = c; sv[0] = f[a]; sv[1] = f[b]; sv[2] = f[c];
}
__device__ __forceinline__ int af_tt(const float* f) {
    int pr[3]; float sv[3]; af_perm(f, pr, sv);
    return 2*pr[0] + (pr[1] > pr[2] ? 1 : 0);
}
__device__ __forceinline__ float af_tc(int t, int ax) {                         // 적분점 (무게중심) 칸 안 격자 좌표
    if (t >= 6) return 0.5f;
    int p0 = t >> 1, r0 = (p0 == 0) ? 1 : 0, r1 = (p0 == 2) ? 1 : 2;
    int p1 = (t & 1) ? r1 : r0;
    return (ax == p0) ? 0.75f : ((ax == p1) ? 0.5f : 0.25f);
}
#define AF_TVW(o) (((o) < 6) ? (1.f/6.f) : 0.f)
__device__ __forceinline__ void af_tw27(const float* f, float* TW, float* TG) {
    for (int o=0;o<27;o++) { TW[o] = 0.f; TG[o*3] = 0.f; TG[o*3+1] = 0.f; TG[o*3+2] = 0.f; }
    int pr[3]; float sv[3]; af_perm(f, pr, sv);
    float lam[4] = {1.f - sv[0], sv[0] - sv[1], sv[1] - sv[2], sv[2]};
    int v[3] = {0, 0, 0};
    for (int r=0;r<4;r++) {
        if (r > 0) v[pr[r-1]] = 1;
        int o = v[0]*9 + v[1]*3 + v[2];
        TW[o] = lam[r];
        float g[3] = {0.f, 0.f, 0.f};
        if (r == 0) g[pr[0]] = -1.f;
        else if (r < 3) { g[pr[r-1]] += 1.f; g[pr[r]] -= 1.f; }
        else g[pr[2]] = 1.f;
        TG[o*3] = g[0]; TG[o*3+1] = g[1]; TG[o*3+2] = g[2];
    }
}
#endif
'''

PERM = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

R_BWF = re.compile(r"for \(int a=0;a<3;a\+\+\) bwf\(fx\[p\*3\+a\], (\w+)\[a\], dw\[a\]\);")
R_BWF3 = re.compile(r"bwf\(fx\[p\*3\], (\w+)\[0\], dw\[0\]\); bwf\(fx\[p\*3\+1\], \1\[1\], dw\[1\]\); bwf\(fx\[p\*3\+2\], \1\[2\], dw\[2\]\);")
NEW_BWF = "float TW_[27], TG_[81]; af_tw27(fx + (size_t)p*3, TW_, TG_);"
R_GRAD = re.compile(r"\{dw\[0\]\[i\]\*(\w+)\[1\]\[j\]\*\1\[2\]\[k\], \1\[0\]\[i\]\*dw\[1\]\[j\]\*\1\[2\]\[k\], \1\[0\]\[i\]\*\1\[1\]\[j\]\*dw\[2\]\[k\]\}")
NEW_GRAD = "{TG_[(i*9+j*3+k)*3], TG_[(i*9+j*3+k)*3+1], TG_[(i*9+j*3+k)*3+2]}"
R_PROD = re.compile(r"(?<![\w])(w|wx)\[0\]\[i\]\*\1\[1\]\[j\]\*\1\[2\]\[k\]")
NEW_PROD = "TW_[i*9+j*3+k]"
R_OC = re.compile(r"\(fx\[([^\]]+)\] >= 1\.f \? 4 : 0\) \+ \(fx\[\1\+1\] >= 1\.f \? 2 : 0\) \+ \(fx\[\1\+2\] >= 1\.f \? 1 : 0\)")
R_TRI = re.compile(r"\(\(oc & 4\) \? t\[0\] : 1\.f - t\[0\]\) \* \(\(oc & 2\) \? t\[1\] : 1\.f - t\[1\]\) \* \(\(oc & 1\) \? t\[2\] : 1\.f - t\[2\]\)")
NEW_TRI = "((oc == af_tt(fx + (size_t)p*3)) ? 1.f : 0.f)"
YQ = [(r"\+ 1\.f \+ \(\(oc & 4\) \? GG : -GG\)", "+ af_tc(q & 7, 0)"), (r"\+ 1\.f \+ \(\(oc & 2\) \? GG : -GG\)", "+ af_tc(q & 7, 1)"),
      (r"\+ 1\.f \+ \(\(oc & 1\) \? GG : -GG\)", "+ af_tc(q & 7, 2)")]

INFO = dict(mode=None, mods=[], err=None)
OUT = ["."]


def _fn_spans(src, name):
    """이름이 name 인 __global__/__device__ 함수 본문 [시작, 끝) 목록."""
    out = []
    for m in re.finditer(r"(?:__global__|__device__)[\w\s]*?void\s+(?:__launch_bounds__\(\d+\)\s+)?" + re.escape(name) + r"\s*\(", src):
        i = src.find("{", m.end())
        if i < 0:
            continue
        d, j = 0, i
        while j < len(src):
            ch = src[j]
            if ch == "{":
                d += 1
            elif ch == "}":
                d -= 1
                if d == 0:
                    break
            j += 1
        if j < len(src):
            out.append((i, j + 1))
    return out


def _sub_fn(src, name, pat, rep, cnt):
    sp = _fn_spans(src, name)
    for a, b in reversed(sp):
        body, n = re.subn(pat, rep, src[a:b])
        cnt[name + ":" + pat[:24]] = cnt.get(name + ":" + pat[:24], 0) + n
        src = src[:a] + body + src[b:]
    return src


def _bodies(src):
    """모든 __global__/__device__ 함수 본문 [시작, 끝) (뒤에서부터 바꾸려고 순서대로)."""
    out = []
    for m in re.finditer(r"(?:__global__|__device__)[\w\s]*?void\s+(?:__launch_bounds__\(\d+\)\s+)?(\w+)\s*\(", src):
        i = src.find("{", m.end())
        if i < 0:
            continue
        d, j = 0, i
        while j < len(src):
            if src[j] == "{":
                d += 1
            elif src[j] == "}":
                d -= 1
                if d == 0:
                    break
            j += 1
        if j < len(src):
            out.append((m.group(1), i, j + 1))
    return out


def transform(src):
    """사면체 기저로 바꾼 소스와 규칙별 적용 수. 대상 글자가 없는 모듈은 그대로.
    입자 가중 (bwf 줄·27 곱·기울기) 은 bwf 를 부르는 함수 본문 안에서만 바꿈 (정답 격자 B-스플라인 커널 -- 예 fem_wolfh5 w5s_acc -- 은 그대로)."""
    cnt = {}
    s = src
    left = []
    for name, a, b in reversed(_bodies(s)):
        body = s[a:b]
        n1 = n2 = 0
        body, n1 = R_BWF.subn(NEW_BWF, body)
        body, n2 = R_BWF3.subn(NEW_BWF, body)
        if not (n1 or n2):
            continue
        body, n3 = R_GRAD.subn(NEW_GRAD, body)
        body, n4 = R_PROD.subn(NEW_PROD, body)
        for k_, v_ in (("bwf", n1), ("bwf3", n2), ("grad", n3), ("prod", n4)):
            if v_:
                cnt[k_] = cnt.get(k_, 0) + v_
        if "bwf(fx" in body or re.search(r"(?<![\w])(w|wx)\[[012]\]\[[ijk]\]", body) or re.search(r"(?<![\w])dw\[[012]\]", body):
            left.append(name)
        s = s[:a] + body + s[b:]
    s, n = R_OC.subn(lambda m_: "af_tt(fx + (size_t)(" + m_.group(1) + "))", s)
    if n:
        cnt["oc"] = n
    s, n = R_TRI.subn(NEW_TRI, s)
    if n:
        cnt["tri"] = n
    for fn in ("rt_coord",):
        s = _sub_fn(s, fn, r"floorf\(y - 0\.5f\)", "floorf(y)", cnt)
    for fn in ("rt_gauss",):
        s = _sub_fn(s, fn, r"ac\[28\]\*0\.125f", "ac[28]*AF_TVW(q & 7)", cnt)
        s = _sub_fn(s, fn, r"ac\[29\]\*0\.125f", "ac[29]*AF_TVW(q & 7)", cnt)
        for p_, r_ in YQ:
            s = _sub_fn(s, fn, p_, r_, cnt)
    s = _sub_fn(s, "gq_pq", r"\]\*0\.125f", "]*AF_TVW(q & 7)", cnt)
    cnt = {k: v for k, v in cnt.items() if v}
    if cnt:
        if "bwf(fx" in s:
            left.append("bwf(fx")
        if re.search(r"fx\[p\*3\] >= 1\.f", s):
            left.append("oc")
        if "? t[0] : 1.f - t[0]" in s:
            left.append("tri")
        if left:
            raise RuntimeError(f"[절 CM102 tet1] 바꾸지 못한 B-스플라인 글자: {left} (규칙 {cnt})")
        k = s.index('extern "C" {') + len('extern "C" {') if 'extern "C" {' in s else 0
        s = s[:k] + DEV + s[k:]
    return s, cnt


def tables(Ai):
    """BW [8*27], BG [8*27*3] (월드 기울기 = 격자 기울기 @ Ai, 원래 표와 같은 꼴)."""
    Ai64 = np.asarray(Ai, dtype=np.float64)
    BW = np.zeros((8, 27)); BG = np.zeros((8, 27, 3))
    for t, (p0, p1, p2) in enumerate(PERM):
        v = [0, 0, 0]; pr = (p0, p1, p2)
        for r in range(4):
            if r > 0:
                v[pr[r - 1]] = 1
            o = v[0] * 9 + v[1] * 3 + v[2]
            g = np.zeros(3)
            if r == 0:
                g[p0] = -1.0
            elif r < 3:
                g[pr[r - 1]] += 1.0; g[pr[r]] -= 1.0
            else:
                g[p2] = 1.0
            BW[t, o] = 0.25
            BG[t, o] = g @ Ai64
    return BW, BG


def check_tables(Ai):
    """표 자체 검산: 분할 1 · 기울기 합 0 · 일차장 재현 (격자 좌표 x = 노드 격자 좌표의 선형장)."""
    BW, BG = tables(Ai)
    A = np.linalg.inv(np.asarray(Ai, dtype=np.float64))
    err = 0.0
    for t, (p0, p1, p2) in enumerate(PERM):
        c = np.zeros(3); c[p0], c[p1], c[p2] = 0.75, 0.5, 0.25
        xs = np.zeros(3); G = np.zeros((3, 3))
        for o in range(27):
            k = np.array([o // 9, (o // 3) % 3, o % 3], dtype=np.float64)
            xs += BW[t, o] * (A @ k); G += np.outer(A @ k, BG[t, o])
        err = max(err, abs(BW[t].sum() - 1), np.abs(BG[t].sum(0)).max(), np.abs(xs - A @ c).max(), np.abs(G - np.eye(3)).max())
    return err


def _fix_tables(s):
    import torch
    if INFO["mode"] != 1 or not hasattr(s, "BW") or not hasattr(s, "BG") or getattr(s, "_tet1_done", False):
        return
    if s.BW.numel() != 216 or s.BG.numel() != 648:
        return
    Ai = getattr(s, "Ai", None)
    Ai = Ai.double().reshape(3, 3).cpu().numpy() if Ai is not None else None
    BW, BG = tables(Ai)
    s.BW.copy_(torch.tensor(BW.reshape(-1), dtype=s.BW.dtype))
    s.BG.copy_(torch.tensor(BG.reshape(-1), dtype=s.BG.dtype))
    s._tet1_done = True
    INFO.setdefault("tables", []).append(dict(cls=type(s).__name__, chk=check_tables(Ai)))
    _dump()


def _dump():
    try:
        json.dump(INFO, open(os.path.join(OUT[0], "tet1info.json"), "w"), ensure_ascii=False, indent=1, default=str)
    except Exception:
        pass


def install(out, mode):
    import cupy as cp
    OUT[0] = out
    INFO["mode"] = int(mode)
    base = cp.RawModule
    if not getattr(base, "_tet1", False):
        def RawModule(*a, **kw):
            a = list(a)
            code = kw.get("code", a[0] if a else None)
            if isinstance(code, str):
                h0 = hashlib.sha1(code.encode()).hexdigest()[:12]
                rec = dict(n=len(INFO["mods"]), sha=h0)
                if INFO["mode"] == 1:
                    try:
                        code2, cnt = transform(code)
                    except Exception as e_:
                        INFO["err"] = str(e_)[:600]; _dump(); raise
                    rec.update(rules=cnt, sha_tet=hashlib.sha1(code2.encode()).hexdigest()[:12])
                    if "code" in kw:
                        kw["code"] = code2
                    else:
                        a[0] = code2
                INFO["mods"].append(rec); _dump()
            return base(*a, **kw)
        RawModule._tet1 = True
        cp.RawModule = RawModule
    from . import fem_an_rt as ART
    mods = [ART]
    for nm in ("fem_an_rt_v0", "fem_ao_rt", "fem_mat_rt"):
        try:
            mods.append(__import__("anchorflow." + nm, fromlist=[nm]))
        except Exception as e_:
            INFO.setdefault("imp_err", []).append(f"{nm}: {e_}"[:200])
    for M in mods:
        for cn in ("RTSolver", "AORTSolver", "RTAdapt", "MatRT"):
            C = getattr(M, cn, None)
            if C is None or getattr(C.__init__, "_tet1", False):
                continue
            f0 = C.__init__

            def init(s, *a, _f0=f0, **kw):
                _f0(s, *a, **kw)
                _fix_tables(s)
            init._tet1 = True
            C.__init__ = init
    print(f"[절 CM102] tet1 켬 -- 모드 {mode} ({'사면체 일차 기저' if mode == 1 else '소스 그대로, 해시만'}), 표 검산 {check_tables(np.linalg.inv(np.eye(3))):.1e}", flush=True)
    _dump()
