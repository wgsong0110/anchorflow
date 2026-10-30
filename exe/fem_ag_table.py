"""절 AG 단독 측정 표: fem_ag_<씬>/ag_<씬>_n<N>_t<회> (GPU0, gpu0.lock, 다른 카드 빈 상태) -- 완주·오차·ms·fps·기준 (≤ 16.7 ms, < 10 %).
사용: python exe/fem_ag_table.py <lbours 루트> <씬...>"""
import collections
import json
import os
import re
import sys

root, scenes = sys.argv[1], sys.argv[2:]


def other_mib(dd):
    v = []
    for f in ("gpu_at_start.txt", "gpu_at_end.txt"):
        p = os.path.join(dd, f)
        if os.path.exists(p):
            for ln in open(p):
                q = [z.strip() for z in ln.split(",")]
                if q[0] != "0":
                    v.append(int(q[1].split()[0]))
    return max(v) if v else None


print("| 씬 | N | 완주 | 오차 (완주 평균; 회별) | ms/프레임 (평균; 회별) | fps | 기준 (≤ 16.7 ms · < 10 %) | 소성 (마지막 회: 손상 입자 / logJp>0 입자) | 다른 카드 최대 MiB |")
print("|---|---|---|---|---|---|---|---|---|")
for sc in scenes:
    d0 = os.path.join(root, f"fem_ag_{sc}")
    G = collections.OrderedDict()
    for d in (sorted(os.listdir(d0)) if os.path.isdir(d0) else []):
        m = re.fullmatch(rf"ag_{sc}_n(\d+)_t(\d+)", d)
        if m:
            G.setdefault(int(m.group(1)), []).append(os.path.join(d0, d))
    for n, ds in G.items():
        S = [json.load(open(os.path.join(d, "summary.json"))) for d in ds if os.path.exists(os.path.join(d, "summary.json"))]
        if not S:
            print(f"| {sc} | {n} | 요약 없음 |")
            continue
        fin = [s for s in S if not s["diverged"]]
        err = sum(s["err_mean"] for s in fin) / len(fin) if fin else None
        ms = sum(s["ms_mean"] for s in S) / len(S)
        ok = err is not None and len(fin) == len(S) and ms <= 1000 / 60 and err < 0.10
        pl = S[-1].get("plast") or {}
        om = max((other_mib(d) or 0) for d in ds)
        print(f"| {sc} | {n} | {len(fin)}/{len(S)} | " + (f"{100 * err:.2f} % (" + ", ".join(f"{100 * s['err_mean_done']:.2f}" for s in S) + ")" if err is not None else "-")
              + f" | {ms:.2f} (" + ", ".join(f"{s['ms_mean']:.2f}" for s in S) + f") | {1000 / ms:.1f} | {'만족' if ok else '미달'} | "
              + f"{pl.get('dmg_final', '-')} / {pl.get('lj_pos_final', '-')} | {om} |")
