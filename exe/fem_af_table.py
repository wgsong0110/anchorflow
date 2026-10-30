"""절 AF 단독 측정 표: af_<궤적>_n<N>_t<회> (GPU0, gpu0.lock) 를 궤적·N 별로 묶어 완주·오차·ms·fps·기준 만족 (60 fps = 16.7 ms 이하, 오차 < 10 %).
다른 카드 상태: 회마다 gpu_at_start/end.txt 의 카드 1~6 메모리 최대 (MiB).
사용: python exe/fem_af_table.py <루트 (…/fem_af_lego)>"""
import collections
import json
import os
import re
import sys

root = sys.argv[1]
G = collections.OrderedDict()
for d in sorted(os.listdir(root)):
    m = re.fullmatch(r"af_(\d{4})_n(\d+)_t(\d+)", d)
    if m:
        G.setdefault((m.group(1), int(m.group(2))), []).append(os.path.join(root, d))


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


print("| 궤적 | N | 완주 | 오차 (완주 평균; 회별) | ms/프레임 (평균; 회별) | fps | 기준 (≤ 16.7 ms · < 10 %) | 다른 카드 최대 MiB |")
print("|---|---|---|---|---|---|---|---|")
ok_all = []
for (t, n), ds in G.items():
    S = [json.load(open(os.path.join(d, "summary.json"))) for d in ds if os.path.exists(os.path.join(d, "summary.json"))]
    if not S:
        print(f"| {t} | {n} | 요약 없음 |")
        continue
    fin = [s for s in S if not s["diverged"]]
    err = sum(s["err_mean"] for s in fin) / len(fin) if fin else None
    ms = sum(s["ms_mean"] for s in S) / len(S)
    good = err is not None and len(fin) == len(ds) and ms <= 1000 / 60 and err < 0.10
    ok_all.append((t, n, good))
    om = max((other_mib(d) or 0) for d in ds)
    print(f"| {t} | {n} | {len(fin)}/{len(ds)} | " + (f"{100 * err:.2f} % " if err is not None else "- ")
          + "(" + ", ".join(f"{100 * (s['err_mean'] if s['err_mean'] is not None else s['err_mean_done']):.2f}" + ("" if not s["diverged"] else f" 발산 {s['diverged']}") for s in S) + ")"
          + f" | {ms:.2f} (" + ", ".join(f"{s['ms_mean']:.2f}" for s in S) + f") | {1000 / ms:.1f} | {'만족' if good else '미달'} | {om} |")
print()
print(f"기준 만족 {sum(g for *_, g in ok_all)}/{len(ok_all)} (궤적·N 조합)")
