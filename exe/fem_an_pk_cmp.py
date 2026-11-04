"""절 AN.15 결과 불변 대조: 그래프 판 / 지속형 판 쌍 (같은 입력) 의 끝 상태 해시·마지막 프레임 위치 차·프레임별 오차 차·시간.
사용: python exe/fem_an_pk_cmp.py <묶음 디렉토리> 기준:대상 ..."""
import json
import os
import sys

import torch

R = sys.argv[1]
print("| 쌍 | 해시 기준 / 대상 | 비트 같음 | 끝 위치 차 평균 / 최대 (m) | 프레임별 오차 차 최대 (%p) | 오차 % 기준 / 대상 | ms 평균 기준 / 대상 | ms 최대 기준 / 대상 | PCG 평균 기준 / 대상 |")
print("|---|---|---|---|---|---|---|---|---|")
for pr in sys.argv[2:]:
    a, b = pr.split(":")
    try:
        sa, sb = [json.load(open(os.path.join(R, q, "summary.json"))) for q in (a, b)]
    except Exception as ex:
        print(f"| {pr} | 없음 ({type(ex).__name__}) |"); continue
    ea = [json.loads(l)["err"] for l in open(os.path.join(R, a, "fem.jsonl"))]
    eb = [json.loads(l)["err"] for l in open(os.path.join(R, b, "fem.jsonl"))]
    de = max(abs(x - y) for x, y in zip(ea, eb)) * 100 if ea and eb else float("nan")
    dx = "-"
    pa, pb = os.path.join(R, a, "pred.pt"), os.path.join(R, b, "pred.pt")
    if os.path.exists(pa) and os.path.exists(pb):
        xa, xb = torch.load(pa)["pred"][-1].double(), torch.load(pb)["pred"][-1].double()
        d = (xa - xb).norm(dim=-1)
        dx = f"{float(d.mean()):.3e} / {float(d.max()):.3e}"
    g = lambda s_, k: (s_.get(k) or {}).get("mean") if isinstance(s_.get(k), dict) else s_.get(k)
    f2 = lambda v: "-" if v is None else f"{v:.2f}"
    print(f"| {pr} | {sa['x_sha1']} / {sb['x_sha1']} | {'예' if sa['x_sha1'] == sb['x_sha1'] else '아니오'} | {dx} | {de:.2e} | {100 * sa['err_mean_done']:.4f} / {100 * sb['err_mean_done']:.4f} | "
          f"{f2(sa['ms_mean'])} / {f2(sb['ms_mean'])} | {f2(sa['ms_max'])} / {f2(sb['ms_max'])} | {f2(g(sa, 'it_nP'))} / {f2(g(sb, 'it_nP'))} |")
for q in sorted(os.listdir(R)):
    p = os.path.join(R, q, "summary.json")
    if os.path.exists(p):
        s = json.load(open(p))
        if s.get("pk"):
            print(q, "지속형", s["pk"])
