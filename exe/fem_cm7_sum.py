"""절 CM7 표: 다시 할당 판별 결과 (평균 오차·완주 프레임·끝 해시·다시 할당 프레임·rc) + 묶음 안 기준 판 (용량 4) 과 대조 (끝 해시 같음, 프레임별 오차 처음 다른 프레임) + memcheck 요약.
사용: python exe/fem_cm7_sum.py <묶음 출력 디렉토리>"""
import glob
import json
import os
import re
import sys

R = sys.argv[1]
REF = {"w1": "w1c4", "w8": "w8c4", "b1": "b1c4", "b8": "b8c4"}            # 묶음별 기준 = 용량 4 배 (다시 할당 없음)


def rd(d):
    o = dict(name=os.path.basename(d).rsplit("_", 1)[0])
    try:
        ri = open(os.path.join(d, "run_info.txt")).read()
        m = re.search(r"rc=(\d+)", ri)
        o["rc"] = m.group(1) if m else "?"
        m = re.search(r"ENV (\S+)", ri)
        o["env"] = m.group(1) if m else "-"
    except Exception:
        o["rc"] = "없음"
    try:
        s = json.load(open(os.path.join(d, "summary.json")))
        o.update(err=s.get("err_mean"), errd=s.get("err_mean_done"), nf=s.get("frames_done"), L=s.get("frames"), h=s.get("x_sha1"), cap=s["args"].get("cap_mul"), K=s["args"].get("K"), ms=s.get("ms_mean"))
    except Exception:
        pass
    try:
        J = [json.loads(q) for q in open(os.path.join(d, "fem.jsonl")) if q.strip()]
        o["errs"] = [j["err"] for j in J]
        o["redo"] = [j["t"] for j in J if j.get("redo")]
        o["nn_max"] = max(j["nn"] for j in J)
    except Exception:
        o["errs"], o["redo"] = [], []
    try:
        lg = open(os.path.join(d, "log.txt"), errors="replace").read()
        m = re.search(r"ERROR SUMMARY: (\d+) error", lg)
        if m:
            o["san"] = m.group(1)
            ks = re.findall(r"at (?:0x[0-9a-f]+ in )?([A-Za-z_][A-Za-z0-9_]*)", lg)
            inv = re.findall(r"Invalid __\w+__ \w+ of size \d+ bytes", lg)
            o["san_k"] = sorted(set(k for k in ks if k.startswith(("rt_", "ao_", "aq_", "mt_", "pk_"))))[:6]
            o["san_i"] = sorted(set(inv))[:3]
        if "illegal memory access" in lg:
            o["ill"] = True
    except Exception:
        pass
    return o


runs = {o["name"]: o for o in (rd(d) for d in sorted(glob.glob(os.path.join(R, "*_*"))) if os.path.isdir(d))}
print("| 판 | 진단 환경 | K | 용량 배수 | rc | 완주 | 평균 오차 % | 끝 해시 | 다시 할당 프레임 (0 부터) | 기준 (용량 4) 과 | memcheck |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for n, o in sorted(runs.items()):
    ref = runs.get(REF.get(n[:2], ""), None)
    cmpv = ""
    if ref and ref is not o and o.get("errs") and ref.get("errs"):
        a, b = o["errs"], ref["errs"]
        k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
        same_h = o.get("h") is not None and o.get("h") == ref.get("h")
        cmpv = ("끝 해시 같음" if same_h else "끝 해시 다름") + (f", 프레임별 오차 비트 같음 ({min(len(a), len(b))} 프레임)" if k is None else f", 프레임 {k} 부터 다름")
    elif ref is o:
        cmpv = "(기준)"
    san = ""
    if "san" in o:
        san = f"오류 {o['san']}" + (f" -- {', '.join(o['san_i'])} @ {', '.join(o['san_k'])}" if o["san"] != "0" else "")
    elif o.get("ill"):
        san = "불법 접근 (memcheck 아님)"
    err = o.get("err")
    errs = f"{100 * err:.3f}" if err is not None else (f"(발산·중단; 끝난 프레임 평균 {100 * o['errd']:.3f})" if o.get("errd") is not None else "-")
    print(f"| {n} | {o.get('env', '-')} | {o.get('K', '-')} | {o.get('cap', '-')} | {o.get('rc')} | {o.get('nf', '-')}/{o.get('L', '-')} | {errs} | {o.get('h', '-')} | {','.join(map(str, o.get('redo', []))) or '없음'} | {cmpv} | {san} |")
