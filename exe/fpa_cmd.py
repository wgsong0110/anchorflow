#!/usr/bin/env python3
"""절 CM190 (final_pa 11-15): 파레토 (우리 판) 와 ablation 의 명령 생성 -- 모든 판이 같은 새 기본 판 위에서 플래그 하나만 다름.
사용: fpa_cmd.py <장면> <판> <반복 번호> <출력 디렉토리>  -> 표준출력에 명령줄.
장면 = L000 (레고) pillow2sofa bread wolf.

새 기본 판 (조정자 11-15) = 장면별 검증된 기본 명령 (exe/pr_ours_tmpl.json: 레고 cm157 LA_1, bread cm157 BA_1, wolf cm139 Kacdhnz;
pillow2sofa = exe/pr2_grid.py 의 유도 (ficus FA_1 명령에서 장면 의존 부분만 + exe/rotbc_wrap.py 구속 = pr2 774b2ed·d29d16c 입자 선택 구속)) 에
  (1) `--cg 4` (절 CM189, 걸음당 PCG 4; 원래 --cg 값을 바꿈)
  (2) 서브스텝 수정 (절 CM184·CM187, substep_fix cmds.md): `--wrx_col 0 --mlow 1e-3` + `--ue_loc 51` + 가장 바깥 감싸개
      `exe/ssfix_wrap.py --ssf_bec 1 --ssf_split 3 --ssf_bigT 0.5 --ssf_tsum 1`.
판 (모두 기본 판에서 하나만 바꿈):
  def          기본 (2x2x2 가우스점, HHT ρ∞ 0.8, 프레임당 1 걸음)
  k2 k4        프레임당 걸음 K 2·4 (파레토 점; 적응 쪼개기 --ksplit 3 --ks_big 은 K 1 전용이라 뺌 -- paper_remeasure 와 같은 규칙)
  q1 q27 q64   가우스점 1·3·4 /축 (exe/q27_wrap.py --q1pt 1, exe/q27b_wrap.py --q27n 3|4)
  be           후퇴 오일러 (--ga be --ue beall, --ue_loc 뺌; lib/anchorflow/fem_an_rt.py 절 CM190)
  mid          중점 (HHT ρ∞ 1 = α 0 = Newmark 평균 가속도, 선형계에서 암시적 중점법과 같음)
  nofpi        격자 -> 입자 F 전달: 가우스점 보간 교체 (--fp_interp) 대신 G2P 누적 F <- (I + Σ U ⊗ ∇w) F
  c075 c15 c20 칸 크기 0.75·1.5·2 배
  ssN          서브스텝 강인성 E: --K N, 50 프레임 (N >= 2 면 --ksplit·--ks_big 뺌)
반복 1 만 덤프 (--dump --dump_F --dump_part) + 풀이기 정보 (q27 0 감싸개, 결과 불변)."""
import json, os, re, sys
scene, var, rep, out = sys.argv[1:5]
rep = int(rep)
HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(f"{HERE}/pr_ours_tmpl.json"))
NW = "/home/dkta/work/newsc"
if scene == "pillow2sofa":                                    # exe/pr2_grid.py NS 유도 그대로
    c = T["ficus"]["cmd"]
    c = c.replace("exe/startpt_wrap.py --sp 1", "exe/rotbc_wrap.py --rot_bc 1 exe/startpt_wrap.py --sp 1", 1)
    c = re.sub(r"AF_SP_FR=1,\d+", "AF_SP_FR=1,300", c)
    c = re.sub(r"--rr_shape \S+", "--rr_shape pillow2sofa", c)
    c = re.sub(r"--rr_traj \S+", f"--rr_traj {NW}/gt_pillow2sofa", c)
    c = re.sub(r"--rr_cfg \S+", f"--rr_cfg {NW}/gt_pillow2sofa/config.json", c)
    c = re.sub(r"--data \S+", f"--data {NW}/pillow2sofa.pt", c)
    c = re.sub(r"--frames \d+", "--frames 300", c)
    c = re.sub(r"--tag \S+", "--tag pillow2sofa", c)
    c = c.replace(" --region_density", "")
else:
    c = T[scene]["cmd"]
c = " ".join(c.split())
c = c.replace(" --dump_part", "").replace(" --dump_F", "").replace(" --dump ", " ")
# ---- 새 기본 판 (진단 판 old = 옛 기본 판 그대로, cgonly = 옛 기본 판 + --cg 4 만: 새 기본 판과 옛 판 차이 가르기)
OLDCMD = var in ("old", "cgonly")
if OLDCMD:
    if var == "cgonly":
        c = re.sub(r" --cg \d+", "", c); c = c.replace(" --newton ", " --cg 4 --newton ", 1)
    if rep == 1:
        c = c.replace(" --K ", " --dump --dump_F --dump_part --K ", 1)
    print(re.sub(r"--out \S+$", f"--out {out}", c)); sys.exit(0)
assert len(re.findall(r" --cg \d+", c)) >= 1
c = re.sub(r" --cg \d+", "", c)
c = c.replace(" --newton ", " --cg 4 --newton ", 1) if " --newton " in c else c
assert c.count(" --cg 4 ") == 1, "--cg 4 자리"
assert c.count(" --wrx_col 1 ") == 1 and c.count(" --ue_loc 50 ") == 1
c = c.replace(" --wrx_col 1 ", " --wrx_col 0 ").replace(" --ue_loc 50 ", " --ue_loc 51 ")
c = re.sub(r" --mlow \S+", "", c)
c = c.replace(" --cell ", " --mlow 1e-3 --cell ", 1)
assert c.count(" --mlow 1e-3 ") == 1
SP = "exe/startpt_wrap.py --sp 1 "
assert c.count("python -u exe/") == 1 and c.count(SP) == 1 and c.count(" exe/fem_an_rt_sim.py ") == 1 and c.count(" --K 1 ") == 1
c = c.replace("python -u exe/", "python -u exe/ssfix_wrap.py --ssf_bec 1 --ssf_split 3 --ssf_bigT 0.5 --ssf_tsum 1 exe/", 1)


def wrap(c, src_cls):   # 가장 안쪽 (실행기 앞, 글자) + 가장 바깥쪽 (startpt 뒤, 클래스) 감싸개 쌍 (abl_gen.py 와 같음)
    return c.replace(" exe/fem_an_rt_sim.py ", f" {src_cls[0]} exe/fem_an_rt_sim.py ").replace(SP, SP + src_cls[1])


if var == "q1":
    c = wrap(c, ("exe/q27_wrap.py --q1pt 1 --q27dg 1 --q27ph src", "exe/q27_wrap.py --q1pt 1 --q27dg 1 --q27ph cls "))
elif var in ("q27", "q64"):
    n = 3 if var == "q27" else 4
    c = wrap(c, (f"exe/q27b_wrap.py --q27 1 --q27n {n} --q27dg 1 --q27ph src", f"exe/q27b_wrap.py --q27 1 --q27n {n} --q27dg 1 --q27ph cls "))
elif rep == 1 and not var.startswith("ss"):   # 풀이기 정보 저장만 (결과 불변), 영 모드 분석용
    c = c.replace(SP, SP + "exe/q27_wrap.py --q27 0 --q27dg 1 ")
if var == "be":
    assert c.count(" --ga hht ") == 1 and c.count(" --ue be ") == 1
    c = c.replace(" --ga hht ", " --ga be ").replace(" --ue be ", " --ue beall ").replace(" --ue_loc 51 ", " ")
elif var == "mid":
    assert c.count(" --rho 0.8 ") == 1 and c.count(" --ga hht ") == 1
    c = c.replace(" --rho 0.8 ", " --rho 1 ")
elif var == "nofpi":
    assert c.count(" --fp_interp ") == 1
    c = c.replace(" --fp_interp ", " ")
m = re.match(r"c(\d+)$", var)
if m:
    mult = {"075": 0.75, "15": 1.5, "20": 2.0}[m.group(1)]
    base = [float(x) for x in re.findall(r"--cell (\S+)", c)][-1]
    c = re.sub(r" --cell \S+", "", c)
    c = c.replace(" --mg_levels ", f" --cell {base * mult:g} --mg_levels ", 1)
m = re.match(r"k(\d+)$", var)
if m and int(m.group(1)) >= 2:
    c = c.replace(" --K 1 ", f" --K {m.group(1)} ", 1).replace(" --ksplit 3", "").replace(" --ks_big", "")
m = re.match(r"ss(\d+)$", var)
if m:
    N = int(m.group(1))
    c = c.replace(" --K 1 ", f" --K {N} ", 1)
    if N >= 2:
        c = c.replace(" --ksplit 3", "").replace(" --ks_big", "")
    c = re.sub(r"--frames \d+", "--frames 50", c); c = re.sub(r"AF_SP_FR=1,\d+", "AF_SP_FR=1,50", c)
if rep == 1 and not var.startswith("ss"):
    c = c.replace(" --K ", " --dump --dump_F --dump_part --K ", 1)
c = re.sub(r"--out \S+$", f"--out {out}", c)
print(c)
