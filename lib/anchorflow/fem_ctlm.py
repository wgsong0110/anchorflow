"""절 CM120 묶음 ctlm (조정자 11-12): 모래 헤시안을 반환 사상과 맞춤 -- 일관 접선 (절 CM108 fem_wolfct --ct 1 의 대칭 몫) 의 비정치를
0 자르기 (PSD 투영) 대신 가우스 점별 Levenberg 이동으로: H_q + μ_q I (F 공간 9 차원, 주값 블록 3 + 비틀림·뒤집기 쌍 6 고유값에 같은 μ_q),
μ_q = max(0, −λ_min(H_q)) -- 그 점 헤시안을 양반정치로 만드는 가장 작은 이동 (Nocedal & Wright 3.4 고유값 이동). 문턱·클램프 없음.
힘·커밋 그대로 (헤시안만). 모래 경로 ao_elem_s 본문 안만; fem_wolfct (ct 1) 의 바꿈 뒤 글자에 덧댐 (설치 순서: 이 모듈 먼저 -> wolfct, RawModule 바깥이 먼저 고침).
"""
ST = dict(n=0)
ANCHOR = "__global__ void __launch_bounds__(128) ao_elem_s("
A_OLD = "float dm[3]; for (int m=0;m<3;m++) { float d_ = fmaxf(D[m][m], 0.f); dm[m] = (d_ > 0.f && isfinite(d_)) ? d_ : 0.f; }"
A_NEW = ("float dm[3]; for (int m=0;m<3;m++) dm[m] = isfinite(D[m][m]) ? D[m][m] : 0.f;   /* 절 CM120 Levenberg 이동 */\n"
         "      float muq = 0.f; for (int m=0;m<3;m++) muq = fmaxf(muq, -dm[m]);\n"
         "      for (int pp=0; pp<3; pp++) { int i, j; ao_pairs(pp, i, j);\n"
         "          float den = ss[i] + ss[j]; float lt = (fabsf(den) > 1e-12f) ? (p[i] + p[j])/den : 0.f;\n"
         "          float dd = ss[i] - ss[j]; float lf = (fabsf(dd) > 1e-6f*(fabsf(ss[i]) + fabsf(ss[j]))) ? (p[i] - p[j])/dd : (Dk[i][i] - Dk[i][j]);\n"
         "          if (isfinite(lt)) muq = fmaxf(muq, -lt); if (isfinite(lf)) muq = fmaxf(muq, -lf); }\n"
         "      for (int m=0;m<3;m++) dm[m] += muq;")
B_OLD = "lt = isfinite(lt) ? fmaxf(lt, 0.f) : 0.f; lf = isfinite(lf) ? fmaxf(lf, 0.f) : 0.f;"
B_NEW = "lt = (isfinite(lt) ? lt : 0.f) + muq; lf = (isfinite(lf) ? lf : 0.f) + muq;   /* 절 CM120 */"


def patch(code):
    if ANCHOR not in code or "절 CM120 Levenberg" in code or "절 CM108 일관 접선" not in code:
        return code
    a_ = code.index(ANCHOR); b_ = code.index("__global__", a_ + len(ANCHOR))
    body = code[a_:b_]
    assert body.count(A_OLD) == 1 and body.count(B_OLD) == 1, ("ctlm 바꿀 자리", body.count(A_OLD), body.count(B_OLD))
    ST["n"] += 1
    return code[:a_] + body.replace(A_OLD, A_NEW).replace(B_OLD, B_NEW) + code[b_:]


def install():
    from . import fem_an_rt as ART
    cpm = ART.cp
    orm = cpm.RawModule

    def RawModule(*a, **kw):
        if "code" in kw:
            kw["code"] = patch(kw["code"])
        elif a:
            a = (patch(a[0]),) + tuple(a[1:])
        return orm(*a, **kw)
    cpm.RawModule = RawModule
    print("[절 CM120] ctlm 켬 -- 일관 접선 대칭 몫 + 가우스 점별 Levenberg 이동 (PSD 투영 대신)", flush=True)
