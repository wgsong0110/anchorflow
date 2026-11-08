"""절 CM72 커널 프로파일 분석: python exe/speed7_kp.py <판 디렉토리> [상위 수] -> <판>/s7kp.json + markdown (stdout).
s7trace.json.gz (fem_speed7 KP, CUDA 활동만) 의 커널·복사를 시뮬 스트림 (요소 커널이 가장 많은 스트림) 과 나머지로 나누고, 프레임 표식 s7_fm 사이를
한 프레임으로. 프레임 안 단계: 앞 (P2G, 첫 걸음 시작 커널 전) · 시작 (걸음 시작 커널 ~ 첫 요소 커널) · 반복 N (N 번째 요소 커널 ~ 다음 요소 커널; 마지막 반복은 판정만)
· 끝 (걸음 끝 커널 ~ ; 걸음 끝 갱신·RM·G2P·렌더 갱신·백업 등). 프레임 무리 = (걸음 시작 수, ao_floor_n 수, 요소 커널 수) -- 가장 흔한 무리 = 보통 프레임.
"""
import collections
import gzip
import json
import os
import sys

STEP0 = {"ao_step0_ga", "rt_step0", "ao_step0_th", "rt_ga_step0", "rt_step0g", "rt_bdf_base"}
ELEM = {"aq_elem", "ao_elem", "ao_elem_s", "rt_elem", "rt_elem_c"}
STEPEND = {"uf_end", "ao_stepend_ga", "rt_stepend", "ao_stepend_th", "rt_stepend_ga"}


def short(n):
    if n.startswith("void ") or "<" in n or "(" in n:
        n = n.replace("void ", "")
        for c in "<(":
            n = n.split(c)[0]
        return n[:48] + "*"
    return n


def analyze(d, top=80):
    fn = os.path.join(d, "s7trace.json.gz")
    ev = json.load(gzip.open(fn))["traceEvents"]
    k = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    k.sort(key=lambda e: e["ts"])
    sc = collections.Counter(e["args"].get("stream") for e in k if e["name"] in ELEM)
    if not sc:
        return None
    sid = sc.most_common(1)[0][0]
    fm = [e for e in k if e["name"] == "s7_fm"]
    frames = []
    for i, m in enumerate(fm):
        t0 = m["ts"]
        t1 = fm[i + 1]["ts"] if i + 1 < len(fm) else None
        frames.append(dict(t0=t0, t1=t1, ev=[]))
    j = 0
    for e in k:
        if e["name"] == "s7_fm":
            continue
        while j + 1 < len(frames) and e["ts"] >= frames[j + 1]["t0"]:
            j += 1
        if frames and e["ts"] >= frames[0]["t0"]:
            frames[j]["ev"].append(e)
    if frames and frames[-1]["t1"] is None:
        frames = frames[:-1]                                                            # 마지막 표식 뒤는 프레임 끝을 모름
    out = []
    for f in frames:
        ph, it, nst, nfl, nel = "앞", -1, 0, 0, 0
        agg = collections.defaultdict(float)
        cnt = collections.Counter()
        oth = collections.defaultdict(float)
        busy = 0.0
        last_end = None
        for e in f["ev"]:
            n = short(e["name"])
            dur = e["dur"] / 1000.0
            if e["args"].get("stream") != sid:
                oth[n] += dur
                continue
            if n in STEP0:
                ph, it = "시작", -1
                nst += 1
            if n == "ao_floor_n":
                nfl += 1
            if n in ELEM:
                it += 1
                nel += 1
                ph = f"반복{it}"
            if n in STEPEND:
                ph = "끝"
            agg[(ph, n)] += dur
            cnt[(ph, n)] += 1
            busy += dur
            last_end = e["ts"] + e["dur"]
        span = (f["t1"] - f["t0"]) / 1000.0
        out.append(dict(sig=(nst, nfl, nel), span=span, busy=busy, agg=agg, cnt=cnt, oth=oth))
    if not out:
        return None
    sig = collections.Counter(q["sig"] for q in out)
    res = dict(n=len(out), sim_stream=sid, groups=[])
    for sg, nn in sig.most_common():
        G = [q for q in out if q["sig"] == sg]
        A = collections.defaultdict(float)
        Cn = collections.defaultdict(float)
        O = collections.defaultdict(float)
        for q in G:
            for kk, v in q["agg"].items():
                A[kk] += v / len(G)
                Cn[kk] += q["cnt"][kk] / len(G)
            for kk, v in q["oth"].items():
                O[kk] += v / len(G)
        ph = collections.defaultdict(float)
        for (p, n), v in A.items():
            ph[p] += v
        kn = collections.defaultdict(float)
        for (p, n), v in A.items():
            kn[n] += v
        res["groups"].append(dict(sig=list(sg), n=nn, span=sum(q["span"] for q in G) / len(G), busy=sum(q["busy"] for q in G) / len(G),
                                  phase=sorted(([p, round(v, 4)] for p, v in ph.items()), key=lambda x: -x[1]),
                                  kern=sorted(([n, round(v, 4)] for n, v in kn.items()), key=lambda x: -x[1])[:top],
                                  pk=sorted(([p, n, round(v, 4), round(Cn[(p, n)], 2)] for (p, n), v in A.items()), key=lambda x: -x[2])[:top],
                                  other=sorted(([n, round(v, 4)] for n, v in O.items()), key=lambda x: -x[1])[:15]))
    return res


def md(d, r, top=40):
    print(f"### 커널 프로파일 {os.path.basename(d)} -- 프레임 {r['n']} (시뮬 스트림 {r['sim_stream']}); 무리 = (걸음 시작 수, ao_floor_n 수, 요소 커널 수)\n")
    for g in r["groups"][:4]:
        print(f"- 무리 {tuple(g['sig'])} {g['n']} 프레임: 표식 사이 {g['span']:.2f} ms, 시뮬 스트림 커널 합 {g['busy']:.2f} ms; 단계 " + " · ".join(f"{p} {v:.2f}" for p, v in g["phase"]))
    g = r["groups"][0]
    print(f"\n| 단계 | 커널 | ms / 프레임 | 부름 / 프레임 |  (가장 흔한 무리 {tuple(g['sig'])}, {g['n']} 프레임)")
    print("|---|---|---|---|")
    for p, n, v, c in g["pk"][:top]:
        print(f"| {p} | {n} | {v:.3f} | {c:.1f} |")
    print("\n| 다른 스트림 | ms / 프레임 |\n|---|---|")
    for n, v in g["other"][:8]:
        print(f"| {n} | {v:.3f} |")
    print()


if __name__ == "__main__":
    d = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    r = analyze(d, top=200)
    if r is None:
        print(f"(커널 프로파일 {d}: 자료 없음)")
        sys.exit(0)
    json.dump(r, open(os.path.join(d, "s7kp.json"), "w"), ensure_ascii=False)
    md(d, r, top)
