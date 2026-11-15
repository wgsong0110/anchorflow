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
// 절 CM180 --tmx 3: 바닥 표본점 노드 64 (사면체 칸 -1 .. +2, 번호 ((a*4+b)*4+c)); 칸 base cb 는 상자 최솟값을 1 낮췄으므로 cb - 1 >= 0
__device__ __forceinline__ void qnodes64(int q, const int* IP, const int* cb, int* ids) {
    int c = q >> 3; int nb1 = IP[7], nb2 = IP[8];
    for (int o=0;o<64;o++) { int i = o/16, j = (o/4)%4, k = o%4; ids[o] = ((cb[c*3]-1+i)*nb1 + (cb[c*3+1]-1+j))*nb2 + (cb[c*3+2]-1+k); }
}
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

# 절 CM180 (사용자 요청 11-15) 혼합 판 `--tmx 1` (`--tmx 2` = P2G 만 B-스플라인, G2P (위치·속도·F 갱신) 은 사면체) (tet1_wrap, 기본 0): 탄성 (적분점·F_q·힘·헤시안) 은 사면체 그대로, 입자 <-> 노드 전달 (P2G·G2P·S_구속 노드 몫) 만 육면체
# 이차 B-스플라인 가중. 노드 = 같은 격자 정수점이라 대응 변환 없음. 사면체 칸 [floor(y), +1) 의 꼭짓점 8 개는 항상 입자의 B-스플라인 노드 (round(y) 둘레 3 칸) 안에 듦.
# 사면체 기저의 fx = y - floor(y) in [0,1) 에서 B-스플라인 base = floor(y - 1/2) = pb + sh (sh = fx < 1/2 ? -1 : 0), 칸 안 좌표 fx - sh.
# 노드 상자는 아래쪽 한 층 늘림 (rt_coord 의 최솟값 b - 1), 위쪽은 사면체 판이 이미 b + 2 까지 가짐.
MIX = [0]
R_PBN = re.compile(r"\(pb\[p\*3(\+1|\+2)?\]\+([ijk])\)")


def _mix_bwf(m_):
    return ("int sh_[3]; for (int a=0;a<3;a++) { sh_[a] = (fx[p*3+a] < 0.5f) ? -1 : 0; bwf(fx[p*3+a] - (float)sh_[a], %s[a], dw[a]); }" % m_.group(1))


def _mix_pb(m_):
    return "(pb[p*3%s]+sh_[%d]+%s)" % (m_.group(1) or "", "ijk".index(m_.group(2)), m_.group(2))


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


def transform(src, mix=False):
    """사면체 기저로 바꾼 소스와 규칙별 적용 수. 대상 글자가 없는 모듈은 그대로.
    입자 가중 (bwf 줄·27 곱·기울기) 은 bwf 를 부르는 함수 본문 안에서만 바꿈 (정답 격자 B-스플라인 커널 -- 예 fem_wolfh5 w5s_acc -- 은 그대로)."""
    cnt = {}
    s = src
    left = []
    for name, a, b in reversed(_bodies(s)):
        body = s[a:b]
        n1 = n2 = 0
        # --tmx 2: G2P 커널 (rt_g2p*) 은 사면체 그대로. e3_wq (--e3w 장벽 표본점 W̄ = 칸 안 입자 G2P 기저 평균, 표 번호 = 사면체 칸 27 노드) 는 --tmx 1·2 에서 사면체 그대로
        # (바닥 처리 사면체), --tmx 3 에서만 B-스플라인 64 노드 판으로 따로 (아래).
        if mix and not (mix == 2 and name.startswith("rt_g2p")) and not (name == "e3_wq" and mix != 3):
            body, n1 = R_BWF.subn(_mix_bwf, body)
            body, n2 = R_BWF3.subn(_mix_bwf, body)
            if not (n1 or n2):
                continue
            body, n3 = R_PBN.subn(_mix_pb, body)
            if name == "e3_wq":                                                         # 표 번호 o (사면체 칸 기준 27) -> B-스플라인 stencil 의 64 (4x4x4, 사면체 칸 -1 .. +2) 번호
                old_ = "atomicAdd(&E3W(q, o), ww[o]*sc);"
                assert body.count(old_) == 1, "e3_wq 쓰기 줄"
                body = body.replace(old_, "{ int i_ = o/9, j_ = (o/3)%3, k_ = o%3; atomicAdd(&E3W(q, ((i_+sh_[0]+1)*4 + (j_+sh_[1]+1))*4 + (k_+sh_[2]+1)), ww[o]*sc); }")
                cnt["e3wq64"] = 1
            cnt["mixbwf"] = cnt.get("mixbwf", 0) + n1 + n2
            cnt["mixpb"] = cnt.get("mixpb", 0) + n3
            if n3 != 3 or re.search(r"pb\[p\*3[^\]]*\]\+[ijk]\b", body) or re.search(r"bwf\(fx\[p\*3\+a\], ", body):
                left.append(name + "(mix pb %d)" % n3)
            s = s[:a] + body + s[b:]
            continue
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
    if mix == 3:                                                                         # 바닥 (장벽 표본점·접선 고착) 을 B-스플라인 W̄ 64 노드로
        for name, a, b in reversed(_bodies(s)):
            body = s[a:b]
            if name in ("e3_wqz", "e3_wqn"):
                body = body.replace("nq*27", "nq*64"); cnt["e3z64"] = cnt.get("e3z64", 0) + 1
            elif ("E3W(" in body or "E3_FLW(" in body) and not name.startswith("e3_"):
                qv = "qq" if re.search(r"\bint\s+qq\b", body) else "q"
                body = body.replace("ids[27]", "ids[64]").replace("qnodes(", "qnodes64(").replace("o<27", "o<64").replace("BW[oc*27+o]", "E3W(%s, o)" % qv)
                if "ids[27]" in body or "o<27" in body or "BW[" in body:
                    left.append(name + "(floor64)")
                cnt["floor64"] = cnt.get("floor64", 0) + 1
            else:
                continue
            s = s[:a] + body + s[b:]
        s = s.replace("BW[(oc)*27+(o)]", "0.f")
        if not cnt.get("floor64") and "E3W(" in s:
            left.append("floor64 없음")
    s, n = R_OC.subn(lambda m_: "af_tt(fx + (size_t)(" + m_.group(1) + "))", s)
    if n:
        cnt["oc"] = n
    s, n = R_TRI.subn(NEW_TRI, s)
    if n:
        cnt["tri"] = n
    for fn in ("rt_coord",):
        s = _sub_fn(s, fn, r"floorf\(y - 0\.5f\)", "floorf(y)", cnt)
        if mix:
            s = _sub_fn(s, fn, r"b0 = b; b1 = b;", "b0 = b - 1; b1 = b;", cnt)
    for fn in ("rt_gauss",):
        s = _sub_fn(s, fn, r"ac\[28\]\*0\.125f", "ac[28]*AF_TVW(q & 7)", cnt)
        s = _sub_fn(s, fn, r"ac\[29\]\*0\.125f", "ac[29]*AF_TVW(q & 7)", cnt)
        for p_, r_ in YQ:
            s = _sub_fn(s, fn, p_, r_, cnt)
    s = _sub_fn(s, "gq_pq", r"\]\*0\.125f", "]*AF_TVW(q & 7)", cnt)
    if os.environ.get("AF_TET1_OCT"):                                                # 옛 실행기 (--gq_mass 없음 -> wmsparse 점유 가중 경로 없음): 사면체 몫 = (1 − κ) 사면체 입자 부피 / 칸 부피 + κ/6
        kap = float(os.environ["AF_TET1_OCT"])
        s = _sub_fn(s, "rt_gauss", r"Vq\[q\] = ac\[28\]\*AF_TVW\(q & 7\); float mm = ac\[29\]\*AF_TVW\(q & 7\);",
                    "float tvw_ = ((q & 7) < 6) ? ((1.f - %.9ef)*aq[27]/ac[28] + %.9ef/6.f) : 0.f; Vq[q] = ac[28]*tvw_; float mm = ac[29]*tvw_;" % (kap, kap), cnt)
    cnt = {k: v for k, v in cnt.items() if v}
    if cnt:
        if "bwf(fx" in s and not mix:
            left.append("bwf(fx")
        if mix and (cnt.get("rt_coord:b0 = b; b1 = b;", 0) != 1 or not cnt.get("mixbwf")):
            left.append("mix 규칙 수 (rt_coord 최솟값 %s, bwf %s)" % (cnt.get("rt_coord:b0 = b; b1 = b;"), cnt.get("mixbwf")))
        if re.search(r"fx\[p\*3\] >= 1\.f", s):
            left.append("oc")
        if "? t[0] : 1.f - t[0]" in s:
            left.append("tri")
        if left:
            raise RuntimeError(f"[절 CM102 tet1] 바꾸지 못한 B-스플라인 글자: {left} (규칙 {cnt})")
        k = s.index('extern "C" {') + len('extern "C" {') if 'extern "C" {' in s else 0
        s = s[:k] + DEV + s[k:]
    return s, cnt


def transform_oct8(src):
    """절 CM102 X9 (사용자 11-10): 이차 B-스플라인 그대로, 옛 실행기 (--gq_mass 없음) 팔분 가우스 점 부피·질량에 점유 가중 κ
    = (1 − κ) 팔분 칸 입자 몫 + κ / 8 (새 실행기 --gq_oct 와 같은 식) -- T9 와 같은 κ 를 쓰기 위함."""
    cnt = {}
    kap = float(os.environ["AF_TET1_OCT"])
    s = _sub_fn(src, "rt_gauss", r"Vq\[q\] = ac\[28\]\*0\.125f; float mm = ac\[29\]\*0\.125f;",
                "float tvw_ = (1.f - %.9ef)*aq[27]/ac[28] + %.9ef/8.f; Vq[q] = ac[28]*tvw_; float mm = ac[29]*tvw_;" % (kap, kap), cnt)
    return s, {k: v for k, v in cnt.items() if v}


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


def install(out, mode, mix=0):
    import cupy as cp
    OUT[0] = out
    MIX[0] = int(mix)
    if int(mix) == 3:
        os.environ["AF_E3K"] = "64"                                                  # fem_escape3: W̄ 표 노드 수 (기본 27)
    INFO["mix"] = int(mix)
    INFO["mode"] = int(mode)
    base = cp.RawModule
    if not getattr(base, "_tet1", False):
        def RawModule(*a, **kw):
            a = list(a)
            code = kw.get("code", a[0] if a else None)
            if isinstance(code, str):
                h0 = hashlib.sha1(code.encode()).hexdigest()[:12]
                rec = dict(n=len(INFO["mods"]), sha=h0)
                if INFO["mode"] == 0 and os.environ.get("AF_TET1_OCT"):                 # X9 (B-스플라인 대조) 수박: 점유 가중 κ 만 (팔분 /8 식)
                    code2, cnt = transform_oct8(code)
                    rec.update(rules=cnt, sha_x9=hashlib.sha1(code2.encode()).hexdigest()[:12])
                    if "code" in kw:
                        kw["code"] = code2
                    else:
                        a[0] = code2
                if INFO["mode"] == 1:
                    try:
                        code2, cnt = transform(code, int(MIX[0]))
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
