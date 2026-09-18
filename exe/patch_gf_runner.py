"""GaussianFluent 최상위 러너의 **수박 전용** 코드를 config 로 켜고 끌 수 있게 한다.

`gs_simulation.py` 는 SH 색으로 수박 껍질 가우시안을 찾아 그 입자들의 beta 를
3e9 로 올린다 (껍질은 안 찢어지게). 다른 물체에는 뜻이 없고, 그 마스크가 **채우기
전** 가우시안 수로 만들어지는데 beta 배열은 **채우기 후** 입자 수라 그대로 터진다 --

    IndexError: size of axis is 143341 but size of corresponding boolean axis is 53606

그래서 config 에 `rind_beta` 가 있을 때만 그 블록을 돌리고, 값도 그 키에서 읽는다.
수박 config 에 `"rind_beta": 3e9` 를 넣으면 원래 동작과 완전히 같다.
"""
from __future__ import annotations

import argparse
import os

MARK = "    # [anchorflow] 수박 껍질 beta 는 config 의 rind_beta 로만 켠다"
OLD = """    beta = mpm_solver.mpm_model.beta.numpy()
    mask1 = (gaussians._features_dc < -0.5).all(axis=2).cpu().numpy().squeeze()"""
NEW = (MARK + """
    import json as _json
    _rind = _json.load(open(args.config)).get("rind_beta", None)
    if _rind is not None:
     beta = mpm_solver.mpm_model.beta.numpy()
     mask1 = (gaussians._features_dc < -0.5).all(axis=2).cpu().numpy().squeeze()""")

ap = argparse.ArgumentParser()
ap.add_argument("--gf", required=True, help="GaussianFluent 체크아웃")
a = ap.parse_args()

p = os.path.join(a.gf, "gs_simulation.py")
s = open(p).read()
if MARK in s:
    raise SystemExit("이미 고쳐져 있다: " + p)
if OLD not in s:
    raise SystemExit("beta 블록을 못 찾았다")
i = s.index(OLD)
j = s.index("    mpm_solver.mpm_model.beta.assign(beta)", i)
body = s[i + len(OLD):j]
# 블록 전체를 한 칸 더 들여쓴다 (if 아래로 들어가므로)
body = "\n".join((" " + ln if ln.strip() else ln) for ln in body.split("\n"))
tail = "     beta[new_mask.cpu().numpy()] = _rind\n     mpm_solver.mpm_model.beta.assign(beta)\n"
body = body.replace(" beta[new_mask.cpu().numpy()] = 3000000000\n", "")
s = s[:i] + NEW + body + tail + s[j + len("    mpm_solver.mpm_model.beta.assign(beta)\n"):]
open(p, "w").write(s)
print("고쳤다:", p, flush=True)
print("PATCH_OK", flush=True)
