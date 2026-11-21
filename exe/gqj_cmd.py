#!/usr/bin/env python3
"""절 CM194 (gq_justify): 가우스 점 판 x 스트레스 조건 명령 생성. 모든 판 = 지금 기본 판 (FFX_ND=2 ffx_cmd <판>.bs2, 절 CM193) 에서 한 축만 바꿈.
사용: gqj_cmd.py <장면> <판> <조건> <모드> <출력 디렉토리>  -> 표준출력 명령줄
  장면  L000 pillow2sofa bread wolf
  판    def (2x2x2) | q27 (3x3x3, exe/q27b_wrap.py 32 자리) | q1 (1x1x1, 같은 8 레인 배치) | q1g001 q1g01 q1g1 (1x1x1 + 벌점 γ 0.01·0.1·1)
  조건  base | e10 e100 (영률 x10·x100, --e_scale) | c15 c20 (칸 x1.5·x2, fpa_cmd) | v2 (충격 x2: 중력 x4·경계 속도 x2) | k2 (프레임당 걸음 2, fpa_cmd k2) | e10c15 e10c20 e100c15 e100c20 (안정 영역 그림용 조합)
  모드  dump (덤프 + 에너지 비) | met (lean + 에너지 비) | time (lean, 진단 없음 = 시간 측정)
"""
import os, re, subprocess, sys
scene, var, cond, mode, out = sys.argv[1:6]
HERE = os.path.dirname(os.path.abspath(__file__))
import re as _re
_m = _re.match(r"(e10|e100)(c15|c20)$", cond)                          # 안정 영역 그림용 두 축 조합 (영률 x 칸)
EC = _m.group(1) if _m else None
base = _m.group(2) if _m else {"base": "def", "e10": "def", "e100": "def", "c15": "c15", "c20": "c20", "v2": "def", "k2": "k2"}[cond]
toks = [base, "bs2"] + (["dump"] if mode == "dump" else ["lean"])
env = dict(os.environ, FFX_ND="2")
c = subprocess.check_output([sys.executable, f"{HERE}/ffx_cmd.py", scene, ".".join(toks), "2", out], env=env, text=True).strip()
c = " " + c + " "
assert c.count(" exe/fem_an_rt_sim.py ") == 1 and c.count("python -u exe/") == 1
if var == "q27":                                         # fpa_cmd 판 q27 과 같은 감싸개 쌍 (가장 안쪽 글자 + 가장 바깥 클래스)
    src_, cls_ = "exe/q27b_wrap.py --q27 1 --q27n 3 --q27dg 0 --q27ph src", "exe/q27b_wrap.py --q27 1 --q27n 3 --q27dg 0 --q27ph cls "
    c = c.replace(" exe/fem_an_rt_sim.py ", f" {src_} exe/fem_an_rt_sim.py ")
    if " exe/startpt_wrap.py --sp 1 " in c:
        c = c.replace(" exe/startpt_wrap.py --sp 1 ", " exe/startpt_wrap.py --sp 1 " + cls_, 1)
    else:                                                  # lean: startpt 없음 -> ssfix 바로 뒤 (ffx_cmd lean 의 q1·q27 배치와 같음)
        m = re.search(r"exe/ssfix_wrap\.py (--ssf_\S+ \S+ )+", c); assert m
        c = c[:m.end()] + cls_ + c[m.end():]
gq = []
if var.startswith("q1"):
    gq += ["--gq_q1 1"]
    g = {"q1": None, "q1g001": "0.01", "q1g01": "0.1", "q1g1": "1"}[var]
    if g:
        gq += [f"--gq_hg {g}"]
else:
    assert var in ("def", "q27"), var
if mode in ("dump", "met"):
    gq += ["--gq_hgm 1"]
if cond == "v2":
    gq += ["--gq_vs 2"]
if cond in ("e10", "e100") or EC:
    assert " --e_scale " not in c
    c = c.replace(" --K ", f" --e_scale {(EC or cond)[1:]} --K ", 1)
if gq:
    c = c.replace("python -u exe/", "python -u exe/gqj_wrap.py " + " ".join(gq) + " exe/", 1)
print(" ".join(c.split()))
