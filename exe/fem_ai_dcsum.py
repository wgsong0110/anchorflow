"""절 AI: fem_dc_cells.py 결과 (dc/dc.json) 를 표 한 줄씩 -- 발산 프레임·서브스텝, 발산 셀 입자 수, 발산 프레임 직전·시작 상태의 그 입자 F (σ3 중앙/최소, σ1/σ3 최대) 우리 vs 정답, 이웃 퍼짐 90 %.
사용: python exe/fem_ai_dcsum.py <dc 실행 디렉토리...>"""
import json
import os
import sys

print("| 판 | 발산 프레임 · 서브스텝 | 발산 셀 입자 (바닥·표면·내부) | 상태 | 우리 σ3 중앙 / 최소 · σ1/σ3 최대 | 정답 σ3 중앙 / 최소 · σ1/σ3 최대 | 퍼짐 90 % 우리 / 정답 |")
print("|---|---|---|---|---|---|---|")
for dd in sys.argv[1:]:
    try:
        d = json.load(open(os.path.join(dd, "dc", "dc.json")))
    except Exception as ex:
        print(f"| {os.path.basename(dd)} | (없음 {ex}) |")
        continue
    lo = d["loc"]
    for h in [q for q in d["hist"] if "gt" in q and "ours" in q][-2:]:
        o, g = h["ours"], h["gt"]
        print(f"| {os.path.basename(dd).replace('dc_', '')} | {d['frame_1based']} · {(d['s_div'] or 0) + 1}/{d['N']} | {lo['n']} ({lo['floor_contact']}·{lo['surface']}·{lo['interior']}) | {h['state']} | "
              f"{o['F']['s3_med']:.3f} / {o['F']['s3_min']:.3f} · {o['F']['ratio_max']:.2f} | {g['F']['s3_med']:.3f} / {g['F']['s3_min']:.3f} · {g['F']['ratio_max']:.2f} | "
              f"{o['spread']['q90']:.2f} / {g['spread']['q90']:.2f} |")
