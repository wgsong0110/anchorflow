"""벤치 결과(회차 디렉토리)를 한 표로 모은다.

  python exe/bench_table.py --run run3 [--md out.md] [--html out.html]

칸마다 모으는 것:
  s(통과 서브스텝) · s_conv(수렴) · ms/프레임 · ms/서브스텝 · FPS ·
  한프레임 오차 e1 · 누적 eT · 입자수 ·
  (있으면) 물리 잔차 중앙값 · 증분 포텐셜 평균 · FID · FVD · KVD
"""
from __future__ import annotations

import argparse
import json
import os

W = "/home/dkta/work"
SHAPES = ["wolf", "mic", "lego", "bread"]
MATS = (os.environ.get("AF_BENCH_MATS")
        or "elastic elastoplastic viscoplastic fracture").split()
MET = ["pg", "ipg", "gasp", "sg"]
NAME = {"pg": "PG", "ipg": "i-PG", "gasp": "GASP", "sg": "Spring-Gaus"}
# 원 논문에 없는 구성식은 **우리가 덧붙여서** 잰다 (exe/sg_materials.py,
# exe/ti_fracture.py). 표에서 그 칸은 "+확장" 으로 표시한다.
EXT = {"gasp": {"fracture": "우리 확장: CD-MPM(Cam-Clay+Borden) 재질 추가"},
       "sg": {"elastoplastic": "우리 확장: 쉬는 길이 소성 갱신",
              "viscoplastic": "우리 확장: 쉬는 길이 점소성 완화",
              "fracture": "우리 확장: 임계 변형률에서 스프링 끊김"}}
NOSUP = {}
MN = {"elastic": "탄성", "elastoplastic": "탄소성",
      "viscoplastic": "점소성", "fracture": "파괴"}

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="run5")
ap.add_argument("--md", default="")
ap.add_argument("--html", default="")
a = ap.parse_args()
O = f"{W}/bench/{a.run}"


def rd(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


rows = []
for m in MET:
    for sh in SHAPES:
        for mt in MATS:
            if mt in NOSUP.get(m, {}):
                rows.append(dict(method=m, shape=sh, material=mt,
                                 note="미지원: " + NOSUP[m][mt]))
                continue
            d = rd(f"{O}/{m}_{sh}_{mt}.json")
            if d is None:
                rows.append(dict(method=m, shape=sh, material=mt,
                                 note="실행 없음"))
                continue
            r = rd(f"{O}/resid/{m}_{sh}_{mt}.json") or {}
            v = rd(f"{O}/vq/{m}_{sh}_{mt}_vq.json") or {}
            rows.append(dict(
                method=m, shape=sh, material=mt,
                n=d.get("n_particles"), s=d.get("s"), s_conv=d.get("s_conv"),
                ms=d.get("ms_per_frame"), mss=d.get("ms_per_substep"),
                fps=d.get("fps"), e1=d.get("e1"), eT=d.get("eT"),
                r_med=r.get("r_med_mean"), ip=r.get("E_mean"),
                fid=v.get("fid"), fvd=v.get("fvd"), kvd=v.get("kvd"),
                note="" if d.get("s") else "기준 미달"))


def f(x, k="{:.2f}"):
    return "-" if x is None else k.format(x)


hdr = ("| 방법 | 형상 | 물성 | 입자 | s | s_conv | ms/프레임 | ms/서브스텝 | "
       "FPS | e1 % | eT % | 잔차 중앙값 % | 증분 포텐셜 | FID | FVD | KVD |")
sep = "|" + "---|" * 16
lines = [hdr, sep]
for q in rows:
    if str(q.get("note", "")).startswith("미지원"):
        lines.append(f"| {NAME[q['method']]} | {q['shape']} | "
                     f"{MN[q['material']]} | " + "미지원 | " * 13
                     + f" <!-- {q['note']} -->")
        continue
    lines.append(
        f"| {NAME[q['method']]} | {q['shape']} | {MN[q['material']]} | "
        f"{f(q.get('n'), '{:d}') if q.get('n') else '-'} | "
        f"{f(q.get('s'), '{:d}') if q.get('s') else '-'} | "
        f"{f(q.get('s_conv'), '{:d}') if q.get('s_conv') else '-'} | "
        f"{f(q.get('ms'))} | {f(q.get('mss'), '{:.3f}')} | {f(q.get('fps'))} | "
        f"{f(None if q.get('e1') is None else 100 * q['e1'], '{:.4f}')} | "
        f"{f(None if q.get('eT') is None else 100 * q['eT'], '{:.3f}')} | "
        f"{f(None if q.get('r_med') is None else 100 * q['r_med'], '{:.4f}')} | "
        f"{f(q.get('ip'), '{:.3e}')} | {f(q.get('fid'), '{:.2f}')} | "
        f"{f(q.get('fvd'), '{:.1f}')} | {f(q.get('kvd'), '{:.5f}')} |"
        + (f"  <!-- {q['note']} -->" if q.get("note") else ""))
md = "\n".join(lines)
print(md)
if a.md:
    open(a.md, "w").write(md + "\n")
    print(f"[저장] {a.md}")
if a.html:
    th = "".join(f"<th>{c.strip()}</th>" for c in hdr.strip("|").split("|"))
    tr = ""
    for q in rows:
        if str(q.get("note", "")).startswith("미지원"):
            cells = [NAME[q["method"]], q["shape"], MN[q["material"]]] + \
                ["미지원"] * 13
            tr += ("<tr style='color:#888'>"
                   + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            continue
        cells = [NAME[q["method"]], q["shape"], MN[q["material"]],
                 f(q.get("n"), "{:d}") if q.get("n") else "-",
                 f(q.get("s"), "{:d}") if q.get("s") else "-",
                 f(q.get("s_conv"), "{:d}") if q.get("s_conv") else "-",
                 f(q.get("ms")), f(q.get("mss"), "{:.3f}"), f(q.get("fps")),
                 f(None if q.get("e1") is None else 100 * q["e1"], "{:.4f}"),
                 f(None if q.get("eT") is None else 100 * q["eT"], "{:.3f}"),
                 f(None if q.get("r_med") is None else 100 * q["r_med"],
                   "{:.4f}"),
                 f(q.get("ip"), "{:.3e}"), f(q.get("fid"), "{:.2f}"),
                 f(q.get("fvd"), "{:.1f}"), f(q.get("kvd"), "{:.5f}")]
        tr += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
    open(a.html, "w").write(
        "<div style='font-family:sans-serif;font-size:12px'>"
        "<table border=1 cellpadding=5 style='border-collapse:collapse'>"
        f"<tr>{th}</tr>{tr}</table></div>")
    print(f"[저장] {a.html}")
json.dump(rows, open(f"{O}/table.json", "w"), indent=1)
