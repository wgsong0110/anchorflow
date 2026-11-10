"""절 CM101 묶음 q27 (사용자 11-10 지시): 요소 적분을 칸마다 가우스 3x3x3 (27 점) 으로 -- 이차 B-스플라인 칸의 정의 그대로 완전 적분.

발견 (조정자, 균일 격자 선형 탄성): 칸마다 2x2x2 적분은 강체 6 개 밖 영 에너지 (모래시계) 모드를 만든다 (3x3x3 칸 덩어리 30 개 대 3x3x3 적분 6 개,
1 칸 144 대 45). 이 모듈은 두 실행기 (새 fem_an_rt · 옛 fem_ao_rt (fem_an_rt_v0 바탕)) 의 고운 층 가우스 점을 27 점으로 바꾼다.

정의 (2x2x2 의 일반화, 새 문턱·클램프 없음):
  - 칸 국소 좌표 u ∈ [0, 1)^3 (칸 가운데 ½). 1 차원 가우스 3 점 ξ = ½ + {−½√(3/5), 0, +½√(3/5)}, 무게 w = {5, 8, 5}/18. 27 점 o = 9 i + 3 j + k,
    무게 W_o = w_i w_j w_k (합 1; 2x2x2 의 ⅛ 자리).
  - 점 몫 칸 (입자 -> 가우스 점, 2x2x2 의 팔분): 축마다 무게 길이로 나눈 세 칸 [0, 5/18) · [5/18, 13/18) · [13/18, 1).
    가우스 점 상태 (F 의 회전·로그 신장 평균, 뒤집힘, 질량·부피, 장벽 표본점, 옛 실행기 logJp·손상 상태) 는 그 몫 칸 입자에서 (2x2x2 와 같은 식).
  - V_q·m_q: 점유 가중 판 (κ = --gq_oct) = (1 − κ) 몫 칸 실제 + κ W_o × 칸 합 (2x2x2 의 '칸 합 / 8' 자리에 W_o), κ < 0 판 = W_o × 칸 합.
  - 보간 교체 (--fp_interp · --lj_interp, 입자 <- 가우스 점): 2x2x2 의 '칸 안 삼선형 (가우스 점 사이 선형, 바깥은 끝 점 값)' 을
    3 점 격자로 -- 축마다 이웃한 두 가우스 점 사이 선형 (u < ½ 이면 ξ0–ξ1, 아니면 ξ1–ξ2), 바깥 (u < ξ0, u > ξ2) 은 끝 점 값 (2x2x2 와 같은 규칙).
  - 거친 멀티그리드 층 (전처리기) 은 2x2x2 그대로 (풀리는 방정식은 고운 층, 전처리기는 근사); 고운 -> 거친 모으기는 27 점 위치로.
구현: 칸당 가우스 점 자리 32 (27 + 빈 자리 5, 부피·질량·기저 0) -- q = 32 c + o. 8 레인 묶음 (seg8·head) 커널은 그대로 (한 칸 = 8 레인 네 묶음, 노드 원자 합이라
  부분 합이 같은 결과). 용량: 칸 용량 capQc 를 4 배로 잡아 (가우스 점 버퍼 capQc×8 = 칸×32) 파이썬 쪽 할당을 그대로 둠, 칸 넘침 검사는 '칸 × 4 > capQc'.
  CUDA 소스는 cp.RawModule 을 감싸 커널 이름별로 고침 (고칠 곳 수를 셈 -- 어긋나면 오류). 커널 목록은 갈무리 (--q27cap) 에서 확인한 고운 층 커널 전부.
"""
import hashlib
import inspect
import json
import os
import sys
import traceback

import numpy as np

ST = dict(mode=0, cap=0, dg=0, exe=None, out=".", nsrc=0, nmod=0, ntr=0, kern={}, err=None, solver=None, nalloc=0)
SQ35 = 0.5 * (0.6 ** 0.5)
XI3 = [0.5 - SQ35, 0.5, 0.5 + SQ35]
W3 = [5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0]


def log(msg):
    print("[절 CM101 q27] " + msg, flush=True)


def _info():
    try:
        with open(os.path.join(ST["out"], "q27info.json"), "w") as f_:
            json.dump({k: v for k, v in ST.items()}, f_, ensure_ascii=False, default=str, indent=1)
    except Exception:
        pass


def _caller():
    for fr in inspect.stack()[2:8]:
        fn = os.path.basename(fr.filename)
        if fn != "fem_q27.py":
            return "%s:%d" % (fn, fr.lineno)
    return "?"


def _capture(code):
    d = os.path.join(ST["out"], "q27src")
    os.makedirs(d, exist_ok=True)
    h = hashlib.sha1(code.encode()).hexdigest()[:10]
    fn = os.path.join(d, "%03d_%s.cu" % (ST["nsrc"], h))
    if not any(x.endswith(h + ".cu") for x in os.listdir(d)):
        with open(fn, "w") as f_:
            f_.write(code)
    with open(os.path.join(d, "manifest.txt"), "a") as f_:
        f_.write("%d %s %s %d\n" % (ST["nsrc"], h, _caller(), len(code)))


def _hook_rawmodule():
    import cupy as cp
    orig = cp.RawModule

    def RawModule(*a, **kw):
        code = kw.get("code") if "code" in kw else (a[0] if a else None)
        if isinstance(code, str):
            ST["nsrc"] += 1
            if ST["cap"]:
                _capture(code)
            if ST["mode"]:
                code2 = transform(code)
                if "code" in kw:
                    kw["code"] = code2
                else:
                    a = (code2,) + tuple(a[1:])
            _info()
        return orig(*a, **kw)

    cp.RawModule = RawModule


def _hook_solvers():
    """첫 풀이기 생성 때 격자 행렬·재질·점유 가중 κ·입자 부피·질량을 q27solver.pt 에 (모드 분석 exe/q27_eig.py 용, 결과 불변)."""
    import torch
    clss = []
    try:
        from . import fem_an_rt as ART
        clss.append(ART.RTSolver)
    except Exception:
        pass
    try:
        from . import fem_ao_rt as AORT
        clss += [AORT.AORTSolver, AORT.RTAdapt]
    except Exception:
        pass
    for C in clss:
        if getattr(C.__init__, "_q27", False):
            continue
        oi = C.__init__

        def init(s, *a, _oi=oi, _C=C, **kw):
            if ST["solver"] is None:
                try:
                    names = ["x", "v", "F", "m", "vol", "A", "Ai", "mu", "lam"]
                    d = {n: a[i] for i, n in enumerate(names) if i < len(a)}
                    sv = dict(cls=_C.__name__, goct=kw.get("goct", None))
                    for n in ("m", "vol", "A", "Ai", "mu", "lam"):
                        v = d.get(n)
                        sv[n] = v.detach().cpu() if torch.is_tensor(v) else v
                    torch.save(sv, os.path.join(ST["out"], "q27solver.pt"))
                    ST["solver"] = dict(cls=_C.__name__, goct=kw.get("goct", None), N=int(d["x"].shape[0]) if "x" in d else None,
                                        A=[float(q) for q in d["A"].reshape(-1).tolist()] if "A" in d and torch.is_tensor(d["A"]) else None,
                                        mu=(float(d["mu"]) if not torch.is_tensor(d.get("mu")) else "tensor"), lam=(float(d["lam"]) if not torch.is_tensor(d.get("lam")) else "tensor"))
                    _info()
                except Exception as e:
                    ST["err"] = "solver_info: " + repr(e)
                    _info()
            return _oi(s, *a, **kw)

        init._q27 = True
        C.__init__ = init


def install(out, mode, cap, dg, exe):
    ST.update(mode=int(mode), cap=int(cap), dg=int(dg), exe=exe, out=out)
    print("[절 CM101] q27 켬 mode %d cap %d dg %d 실행기 %s" % (ST["mode"], ST["cap"], ST["dg"], exe), flush=True)
    try:
        _hook_rawmodule()
        _hook_solvers()
        if ST["mode"]:
            _install_q27()
    except Exception as e:
        ST["err"] = repr(e) + " " + traceback.format_exc()[-1500:]
        _info()
        raise
    _info()


# ---------------------------------------------------------------- 27 점 판 (mode 1)
import re

Q27_HELP = r"""
// ---- 절 CM101 q27: 칸마다 가우스 3x3x3 (lib/anchorflow/fem_q27.py). 점 o = 9 i + 3 j + k (o < 27), 칸당 자리 32 (27..31 빈 자리: 무게 0)
__device__ __constant__ float Q27D[3] = {-0.38729833462074170f, 0.f, 0.38729833462074170f};
__device__ __constant__ float Q27W[32] = {%s};
__device__ __forceinline__ int q27_i(float f) { float u = f - 0.5f; return (u < 0.27777777777777778f) ? 0 : ((u < 0.72222222222222222f) ? 1 : 2); }
__device__ __forceinline__ int q27_oc(float f0, float f1, float f2) { return q27_i(f0)*9 + q27_i(f1)*3 + q27_i(f2); }
__device__ __forceinline__ float q27_d(int oc, int a) { if (oc >= 27) return 0.f; int i = (a == 0) ? oc/9 : ((a == 1) ? (oc/3)%%3 : oc%%3); return Q27D[i]; }
// 보간 교체 축 몫: 이웃한 두 가우스 점 사이 선형 (u < ½ 이면 점 0–1, 아니면 1–2), 바깥은 끝 점 값 (2x2x2 판의 같은 규칙)
__device__ __forceinline__ void q27_ax(float f, int* lo, float* t) { float u = f - 0.5f; const float x0 = 0.11270166537925831f, x2 = 0.88729833462074169f;
    if (u < 0.5f) { *lo = 0; *t = fminf(fmaxf((u - x0)/(0.5f - x0), 0.f), 1.f); } else { *lo = 1; *t = fminf(fmaxf((u - 0.5f)/(x2 - 0.5f), 0.f), 1.f); } }
"""


def _w32():
    w = []
    for o in range(32):
        if o < 27:
            w.append(W3[o // 9] * W3[(o // 3) % 3] * W3[o % 3])
        else:
            w.append(0.0)
    return w


def _help():
    return Q27_HELP % ", ".join("%.17ef" % v for v in _w32())


# 함수 이름 분류. COARSE = 거친 층·결정적 판 전용·노드 커널 (고치지 않음), SPECIAL = 아래 따로 고침 (+ 일반 규칙), 나머지 고운 층 = 일반 규칙.
COARSE = set("""seg8f seg8d rs_fin rt_vn rt_step0 rt_g2p_end rt_mg_dims rt_mg_fin qnodes_d rt_mg_elem rt_mg_mv rt_mg_restrict rt_mg_prolong rt_zero_lm
rt_mg_D rt_mg_node rt_mg_pw1 rt_mg_pw2 rt_fill_rand rt_mg_csolve rt_mg_mv2 rt_mg_restrict2 rt_mg_restrict3 rt_mg_occ rt_mg_cvc cvc_Kz rt_mg_prolong_c
rt_bar_mvl rt_mg_dbar ao_mg_gath ao_mg_acc_det ao_mg_cell mv2_core""".split())
DETONLY = set("ao_mg_gath ao_mg_acc_det ao_mg_cell".split())
PAT = re.compile(r'(>>\s*3\b|&\s*7\b|\*\s*8\b|\b8\s*\*|0\.125|\bGG\b|>= 1\.f \?|oc\s*<\s*8|o\s*<\s*8;)')
GEN = [
    (re.compile(r'\b(qq|q)\s*>>\s*3\b'), r'\1 >> 5'),
    (re.compile(r'\b(qq|q)\s*&\s*7\b'), r'\1 & 31'),
    (re.compile(r'\bc\s*\*\s*8\s*\+\s*oc\b'), 'c*32 + oc'),
    (re.compile(r'\(\s*([^()?]+?)\s*>=\s*1\.f\s*\?\s*4\s*:\s*0\s*\)\s*\+\s*\(\s*([^()?]+?)\s*>=\s*1\.f\s*\?\s*2\s*:\s*0\s*\)\s*\+\s*\(\s*([^()?]+?)\s*>=\s*1\.f\s*\?\s*1\s*:\s*0\s*\)'),
     r'q27_oc(\1, \2, \3)'),
    (re.compile(r'\(\(oc & 4\) \? GG : -GG\)'), 'q27_d(oc, 0)'),
    (re.compile(r'\(\(oc & 2\) \? GG : -GG\)'), 'q27_d(oc, 1)'),
    (re.compile(r'\(\(oc & 1\) \? GG : -GG\)'), 'q27_d(oc, 2)'),
    (re.compile(r'BG \+ \(j\*27\+o\)\*3'), 'BG + ((((qq) & 24) + j)*27+o)*3'),
    (re.compile(r'IP\[12\] = carry\*8; if \(carry > capQc\)'), 'IP[12] = carry*32; if (carry*4 > capQc)'),
]
# 일반 규칙 뒤에도 남아도 되는 글자 (함수별; 8 레인 묶음·노드 27·장벽 8 꼭짓점 등)
OKLEFT = [
    re.compile(r'threadIdx\.x\s*&\s*7'), re.compile(r'\btl\s*&\s*7'), re.compile(r'tl\s*&\s*~7'), re.compile(r'\btl\s*>>\s*3'), re.compile(r'threadIdx\.x\s*>>\s*3'),
    re.compile(r'blockDim\.x\s*>>\s*3'), re.compile(r'\bcn\s*>>'), re.compile(r'cn\s*<\s*8'), re.compile(r'\(cn>>2\)&1'), re.compile(r'j\s*<\s*8;'),
    re.compile(r'r\s*\*\s*8'), re.compile(r'8\s*\*\s*r\b'), re.compile(r'oc8|o8_'), re.compile(r'const float GG = '), re.compile(r'oc<8; oc\+\+\) \{ const float\* Qo = sw'), re.compile(r'LM\s*\+\s*l\*8|l\*8\s*\+|\(l-1\)\*8|\(lf\+1\)\*8|lf\*8'),
]


def _split(code):
    """(머리, [(이름, 본문)]): __global__/__device__ 로 시작하는 줄마다 끊음 (다음 정의 앞까지가 본문)."""
    idx = []
    for m in re.finditer(r'^(?:__global__|__device__)[^\n]*', code, re.M):
        h = re.sub(r'__launch_bounds__\(\s*\d+(\s*,\s*\d+)?\s*\)', '', m.group(0))
        mm = re.search(r'(\w+)\s*\(', h)
        idx.append((m.start(), mm.group(1) if mm else None))
    if not idx:
        return code, []
    head = code[:idx[0][0]]
    parts = []
    for i, (st, n) in enumerate(idx):
        en = idx[i + 1][0] if i + 1 < len(idx) else len(code)
        parts.append((n, code[st:en]))
    return head, parts


def _sub(b, old, new, n=1, name=""):
    c = b.count(old)
    if c != n:
        raise RuntimeError("q27 고침 '%s' 수 %d (기대 %d) in %s" % (old[:80], c, n, name))
    return b.replace(old, new)


def _special(n, b):
    if n == "rt_gauss":
        b = _sub(b, "ac[28]*0.125f", "ac[28]*Q27W[oc]", 1, n)
        b = _sub(b, "ac[29]*0.125f", "ac[29]*Q27W[oc]", 1, n)
        b = _sub(b, "for (int k=0;k<9;k++) Fq[(size_t)q*9+k] = out[k];",
                 "if (oc >= 27) for (int k=0;k<9;k++) out[k] = (k % 4 == 0) ? 1.f : 0.f;   // 절 CM101 빈 자리\n    for (int k=0;k<9;k++) Fq[(size_t)q*9+k] = out[k];", 1, n)
    elif n == "gq_pq":
        b = _sub(b, "PC[(size_t)c*6+a]*0.125f", "PC[(size_t)c*6+a]*Q27W[q & 31]", 1, n)
    elif n == "rt_mg_acc":                                                          # 고운 층 (l == 1) 출처만 27 점; 거친 층 팔분 (qc) 그대로
        b = _sub(b, "int c = q >> 3, oc = q & 7;", "int c = (l == 1) ? (q >> 5) : (q >> 3), oc = (l == 1) ? (q & 31) : (q & 7);", 1, n)
        b = _sub(b, "float tq[3] = {cx + 1.f + ((oc & 4) ? GG : -GG), cy + 1.f + ((oc & 2) ? GG : -GG), cz + 1.f + ((oc & 1) ? GG : -GG)};",
                 "float tq[3] = {cx + 1.f + ((l == 1) ? q27_d(oc, 0) : ((oc & 4) ? GG : -GG)), cy + 1.f + ((l == 1) ? q27_d(oc, 1) : ((oc & 2) ? GG : -GG)), cz + 1.f + ((l == 1) ? q27_d(oc, 2) : ((oc & 1) ? GG : -GG))};", 1, n)
        return b, False                                                             # 일반 규칙 안 씀 (거친 층 qc 팔분)
    elif n == "rt_mv2":
        b = _sub(b, "mv2_core(j, on,", "mv2_core_q27((int)(qq & 24), j, on,", 1, n)
    elif n == "rt_mv2s":
        b = _sub(b, "const float* bq = BG + j*81;", "const float* bq = BG + ((qq & 24) + j)*81;", 1, n)
        b = _sub(b, "const float* bg = BG + oc*81 + o*3;", "const float* bg = BG + ((qq & 24) + oc)*81 + o*3;", 1, n)
    elif n in ("fpi_p", "lji_p"):
        b = _sub(b, "const float GG = 0.28867513459481287f;\n    float t[3]; for (int a=0;a<3;a++) t[a] = fminf(fmaxf((fx[p*3+a] - (1.f - GG)) / (2.f*GG), 0.f), 1.f);",
                 "int lo_[3]; float t[3]; for (int a=0;a<3;a++) q27_ax(fx[p*3+a], &lo_[a], &t[a]);   // 절 CM101 27 점 보간", 1, n)
        b = _sub(b, "for (int oc=0; oc<8; oc++) {\n        float w = ((oc & 4) ? t[0] : 1.f - t[0]) * ((oc & 2) ? t[1] : 1.f - t[1]) * ((oc & 1) ? t[2] : 1.f - t[2]);",
                 "for (int o8_=0; o8_<8; o8_++) {\n        float w = ((o8_ & 4) ? t[0] : 1.f - t[0]) * ((o8_ & 2) ? t[1] : 1.f - t[1]) * ((o8_ & 1) ? t[2] : 1.f - t[2]);\n"
                 "        int oc = (lo_[0] + ((o8_ >> 2) & 1))*9 + (lo_[1] + ((o8_ >> 1) & 1))*3 + (lo_[2] + (o8_ & 1));", 1, n)
    elif n == "aq_p2g_sd":
        b = _sub(b, "float a[8][4]; for (int o=0;o<8;o++)", "float a[32][4]; for (int o=0;o<32;o++)", 1, n)
        b = _sub(b, "for (int o=0;o<8;o++) for (int k=0;k<4;k++) Sa[((size_t)c*8+o)*4+k] = a[o][k];", "for (int o=0;o<32;o++) for (int k=0;k<4;k++) Sa[((size_t)c*32+o)*4+k] = a[o][k];", 1, n)
    elif n in ("ao_gath_q", "gq_gath"):
        b = re.sub(r'for \(int oc=0;oc<8;oc\+\+\)', 'for (int oc=0;oc<32;oc++)', b)
        b = b.replace("qv[(size_t)c*8+oc]", "qv[(size_t)c*32+oc]")
    elif n in ("sp_csum", "sp_q2n"):
        b = re.sub(r'for \(int oc=0; oc<8; oc\+\+\)', 'for (int oc=0; oc<32; oc++)', b)
    return b, True


def _mv2_core_q27(b):
    b = _sub(b, "__device__ __forceinline__ void mv2_core(int j,", "__device__ __forceinline__ void mv2_core_q27(int gofs, int j,", 1, "mv2_core")
    b = _sub(b, "const float* bq = BG + j*81;", "const float* bq = BG + (gofs + j)*81;", 1, "mv2_core")
    b = _sub(b, "const float* bg = BG + oc*81 + o*3;", "const float* bg = BG + (gofs + oc)*81 + o*3;", 1, "mv2_core")
    return b


def transform(code):
    head, parts = _split(code)
    if not parts or "IP[12]" not in code:
        return code
    out, rep, unk = [], {}, []
    for n, b in parts:
        if n is None or not PAT.search(b) or n in COARSE or (n.startswith(("rt_mg_", "ao_mg_", "cvc")) and n != "rt_mg_acc"):
            out.append(b)
            if n == "mv2_core" and "mv2_core(j, on," in code:
                out.append(_mv2_core_q27(b))
            continue
        b2, gen = _special(n, b)
        if gen:
            for rx, rp in GEN:
                b2 = rx.sub(rp, b2)
        left = []
        for ln in b2.split("\n"):
            core = ln.split("//")[0]
            for o_ in OKLEFT:
                core = o_.sub(" ", core)
            if PAT.search(core) and n != "rt_mg_acc":
                left.append(ln.strip()[:160])
        if left:
            unk.append((n, left))
        rep[n] = int(b2 != b)
        out.append(b2)
    ST["kern"].update(rep)
    if unk:
        ST["left"] = ST.get("left", []) + [(n, l[:4]) for n, l in unk]
    new = head + "".join(out)
    i = new.find('extern "C" {')
    new = (new[:i + len('extern "C" {')] + _help() + new[i + len('extern "C" {'):]) if i >= 0 else _help() + new
    ST["ntr"] += 1
    return new


def tables(Ai):
    """27 점 기저 표 (자리 32, 빈 자리 0): BW [32*27], BG [32*27*3] (월드 기울기 = 격자 기울기 @ Ai), 2x2x2 판과 같은 식."""
    Ai64 = Ai.double().cpu().numpy().reshape(3, 3)
    BW = np.zeros((32, 27)); BG = np.zeros((32, 27, 3))
    for oc in range(27):
        ii = (oc // 9, (oc // 3) % 3, oc % 3)
        fxv = [1.0 + (XI3[ii[a]] - 0.5) for a in range(3)]
        w = [[0.5 * (1.5 - f) ** 2, 0.75 - (f - 1) ** 2, 0.5 * (f - 0.5) ** 2] for f in fxv]
        dw = [[f - 1.5, -2 * (f - 1), f - 0.5] for f in fxv]
        for o in range(27):
            i, j, k = o // 9, (o // 3) % 3, o % 3
            BW[oc, o] = w[0][i] * w[1][j] * w[2][k]
            gy = np.array([dw[0][i] * w[1][j] * w[2][k], w[0][i] * dw[1][j] * w[2][k], w[0][i] * w[1][j] * dw[2][k]])
            BG[oc, o] = gy @ Ai64
    return BW, BG


COARSE_K = ("rt_mg_", "ao_mg_", "cvc")


def _install_q27():
    import torch
    from . import fem_an_rt as ART
    bases = [ART.RTSolver]
    try:
        from . import fem_an_rt_v0 as V0
        bases.append(V0.RTSolver)
    except Exception:
        pass
    P = ART.P
    for C in bases:
        if C.__dict__.get("_q27", False):
            continue
        C._q27 = True
        oa, oi, ok = C.alloc, C.__init__, C.k

        def alloc(s, need, _oa=oa):
            if need is None:                                                         # 원래 식 그대로 + 칸 수 x 4 (가우스 점 버퍼 capQc x 8 = 칸 x 32)
                y = s.x.double() @ s.Ai.double().T
                b = torch.floor(y - 0.5)
                nb = (b.max(0).values - b.min(0).values + 3).long().tolist()
                nn = nb[0] * nb[1] * nb[2]
                ncb = (nb[0] - 2) * (nb[1] - 2) * (nb[2] - 2)
                uc = torch.unique(((b - b.min(0).values).long() * torch.tensor([(nb[1] - 2) * (nb[2] - 2), nb[2] - 2, 1], device=s.dev)).sum(1)).numel()
                need = (nn, ncb, 4 * uc)
                s.nb0 = nb
                ST["nalloc"] += 1
                ST["alloc_last"] = dict(nn=nn, ncb=ncb, uc=uc)
            else:
                raise RuntimeError("q27: alloc(need) 경로는 지원 안 함")
            _oa(s, need)
            ST["capQc"] = int(s.capQc)
            _info()

        def init(s, *a, _oi=oi, **kw):
            _oi(s, *a, **kw)
            if getattr(s, "_q27tab", False):
                return
            s._q27tab = True
            BW, BG = tables(s.Ai)
            s.BW8, s.BG8 = s.BW, s.BG                                               # 거친 층 (전처리기) 은 2x2x2 표
            s.BW = torch.tensor(BW, device=s.BG8.device, dtype=torch.float32).reshape(-1).contiguous()
            s.BG = torch.tensor(BG, device=s.BG8.device, dtype=torch.float32).reshape(-1).contiguous()
            ST["tab"] = dict(bw_rowsum_min=float(BW[:27].sum(1).min()), bw_rowsum_max=float(BW[:27].sum(1).max()), bg_rowsum_absmax=float(np.abs(BG[:27].sum(1)).max()),
                             w_sum=float(sum(_w32())), cls=type(s).__name__)
            _info()

        def k(s, name, n, args, b=128, _ok=ok):
            if name.startswith(COARSE_K) and getattr(s, "BG8", None) is not None:
                pb, pw = P(s.BG), P(s.BW)
                args = tuple((P(s.BG8) if (isinstance(x, np.uint64) and x == pb) else (P(s.BW8) if (isinstance(x, np.uint64) and x == pw) else x)) for x in args)
                ST["ncoarse"] = ST.get("ncoarse", 0) + 1
            return _ok(s, name, n, args, b)

        C.alloc, C.__init__, C.k = alloc, init, k
    # 프레임 점검 (결과 불변, 그래프 밖): 풀이 함수 들어갈 때 앞 프레임 가우스 점 합 -- Σ V_q = Σ 입자 부피, Σ m_q = Σ 입자 질량, 빈 자리 V·m = 0
    mods = [ART]
    try:
        from . import fem_ao_rt as AORT
        mods.append(AORT)
    except Exception:
        pass
    for M in mods:
        for fn in ("split_step", "adapt_step"):
            f0 = getattr(M, fn, None)
            if f0 is None or getattr(f0, "_q27", False):
                continue

            def w(s, *a, _f0=f0, **kw):
                try:
                    _check(s)
                except Exception as e:
                    ST["chk_err"] = repr(e)
                return _f0(s, *a, **kw)

            w._q27 = True
            setattr(M, fn, w)
    ST["installed"] = True
    _info()


_NCHK = [0]


def _check(s):
    import torch
    _NCHK[0] += 1
    i = _NCHK[0]
    if not (i <= 6 or i % 10 == 0) or not ST["dg"]:
        return
    with torch.no_grad():
        torch.cuda.synchronize()
        nq, nc = int(s.IP[12]), int(s.IP[11])
        if nq <= 0:
            return
        V = s.Vq[:nq].double(); m = s.mq[:nq].double()
        sl = torch.arange(nq, device=V.device) & 31
        dmy = sl >= 27
        r = dict(call=i, nq=nq, nc=nc, nq_ok=int(nq == 32 * nc), sumV=float(V.sum()), sumvol=float(s.vol.double().sum()), summ=float(m.sum()), summp=float(s.m.double().sum()),
                 dummyV=float(V[dmy].abs().sum()), dummym=float(m[dmy].abs().sum()), nV0=int((V[~dmy] <= 0).sum()), nreal=int((~dmy).sum()),
                 Fdet_min=float(torch.linalg.det(s.Fq[:nq][~dmy].double().reshape(-1, 3, 3)).min()))
        with open(os.path.join(ST["out"], "q27.jsonl"), "a") as f_:
            f_.write(json.dumps(r) + "\n")
