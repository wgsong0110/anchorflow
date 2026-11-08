"""절 CM65 묶음 speed6 프로파일 요약: python exe/speed6_prof.py <결과 디렉토리> <판...> -> markdown (stdout).
s6prof.json (fem_speed6.analyze) 의 'GPU 가 빈 동안 호스트 자기 몫' 상위 항목을 갈래로 묶음 (프로파일러가 파이썬 호출을 다 기록해 호스트가 느려지므로 몫의 크기 순서만 봄).
"""
import json
import os
import statistics as stt
import sys

CAT = [
    ("E1 진단 (escape1 diag)", ("escape1_wrap.py(78): diag",)),
    ("기록 훅 읽기 (speed3·t2chain·speed4 s3/t2c/s4.jsonl)", ("frame_rec", "ws4_counts", "yl4_counts", "x3e_frame_n", "inv_counts", "fem_speed3.py(398): w <-", "fem_speed4.py(823): w <-")),
    ("H5 기록 읽기 (hhtc2 hc.jsonl)", ("fem_hhtc2.py(275): step <-",)),
    ("프레임 시작 상태 백업 (_bk clone)", ("_bk",)),
    ("쪼개기 판정 읽기 (nfail)", ("nfail",)),
    ("렌더 호출 (갱신 띄우기·래스터화 전처리 개수 읽기·동기화)", ("rtrender1", "rasterize", "diff_gaussian", "upd_pg", "upd_gf", "ras_pg", "ras_gf", "fem_speed5.py(2", "_frame_r", "r_flush")),
    ("그래프 띄우기·구간 판정 읽기 (replay·CT[0])", ("replay",)),
    ("구속 갱신 (set_bc)", ("set_bc",)),
]


def cat(k):
    if k.startswith("밖 "):
        return "실행기 프레임 사이 (평가·기록·진단)"
    for name, keys in CAT:
        if any(q in k for q in keys):
            return name
    return "기타 (torch 호출 겹·파이썬)"


O = sys.argv[1]
for n in sys.argv[2:]:
    p = os.path.join(O, n, "s6prof.json")
    if not os.path.exists(p):
        continue
    r = json.load(open(p))
    F = r["frames"]
    idle_in = stt.mean(q["idle_in_ms"] for q in F)
    idle_out = stt.mean(q["idle_out_ms"] for q in F)
    agg = {}
    for k, v in r["self_per_frame"]:
        c = cat(k)
        agg[c] = agg.get(c, 0.0) + v
    tot = sum(agg.values())
    print(f"### {n}: 프레임 {len(F)} · GPU 빈 시간 쪼개기 안 {idle_in:.2f} / 프레임 사이 {idle_out:.2f} ms (프로파일러 켬)\n")
    print("| 갈래 | ms / 프레임 |")
    print("|---|---|")
    for c, v in sorted(agg.items(), key=lambda q: -q[1]):
        print(f"| {c} | {v:.3f} |")
    print(f"| (상위 {len(r['self_per_frame'])} 항목 밖) | {max(0.0, idle_in + idle_out - tot):.3f} |\n")
