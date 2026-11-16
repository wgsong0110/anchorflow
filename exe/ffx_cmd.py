#!/usr/bin/env python3
"""절 CM191 (final_fix 11-15): --cg 4 회귀 진단·공통 규칙 찾기 명령 생성.
사용: ffx_cmd.py <장면> <판> <반복> <출력 디렉토리>  -> 표준출력에 명령줄.
장면:
  L000 pillow2sofa bread wolf  -> exe/fpa_cmd.py (절 CM190) 의 판 그대로 (판 이름 첫 토막 = fpa_cmd 판: def old k2 k4 be mid nofpi q1 ...)
  chair_paste_lego drums_honey_lego ship_plastic_lego bread_foam_drop_lego
                               -> 절 CM189 레고 세팅 점소성 장면: exe/nshape_cmd.py (BA_1 명령) + substep_fix (vast_plastfix_lego.sh run1 과 같음),
                                  단독 시간 판 (진단 플래그 뺌 NS_DROP, 덤프 없음). 첫 토막 def = NS_CG4 1 + ssfix, old = NS_CG4 0 + ssfix 없음.
판 = 첫 토막 + '.' 로 이은 바꿈 토막 (차례대로 적용):
  cgN      --cg N               tpX      --tolP X (X 는 점 없이, 예 tp3e-1)              nofpi    --fp_interp 뺌
  ssw      ssfix 바깥 감싸개 넣음 (old 에)                       mlow / nomlow   --mlow 1e-3 넣음 / 뺌
  uelN     --ue_loc N           wcN      --wrx_col N           dump     --dump --dump_F --dump_part
  pfx      --fp_interp 넣음 (없으면)   fel  --fpi_el 1 (--fp_interp 탄성 재질에만)   wrxN  --wrx N
새 기본 판 후보 이름은 절 CM191 이 정하면 여기에 별칭으로 둔다 (NEW)."""
import json, os, re, subprocess, sys
scene, var, rep, out = sys.argv[1:5]
HERE = os.path.dirname(os.path.abspath(__file__))
NEW = {}                                               # 별칭: 새 기본 판 (절 CM191 에서 채움)
toks = NEW.get(var, var).split(".")
base, mods = toks[0], toks[1:]
SSW = "exe/ssfix_wrap.py --ssf_bec 1 --ssf_split 3 --ssf_bigT 0.5 --ssf_tsum 1 "
NSH = {"chair_paste_lego", "drums_honey_lego", "ship_plastic_lego", "bread_foam_drop_lego", "hotdog_plastic_lego"}
if scene in NSH:
    W = "/home/dkta/work"; AG = f"/dev/shm/dkta/ag_{scene}"
    NF = json.load(open(f"{AG}/gt_{scene}/config.json"))["frame_num"]
    env = dict(os.environ, NS_CG4="1" if base == "def" else "0", NS_NODUMP="1", NS_DROP="--pdiag --qjd --fhash --hgt --bar_cnt")
    assert base in ("def", "old")
    c = subprocess.check_output([sys.executable, f"{HERE}/nshape_cmd.py", f"{W}/BA_1.txt", f"{AG}/{scene}.pt", str(NF), scene, out, f"{AG}/gt_{scene}", f"{AG}/gt_{scene}/config.json", "1"], env=env, text=True).strip()
    if base == "def":
        assert "--mlow 1e-3" in c and c.count("python -u exe/") == 1
        c = c.replace("python -u exe/", "python -u " + SSW + "exe/", 1).replace("--ue_loc 50 ", "--ue_loc 51 ").replace("--wrx_col 1 ", "--wrx_col 0 ")
elif os.environ.get("FFX_ND", "0") == "1":              # 절 CM191 새 기본 판 = CM190 기본 판 + --fpi_el 1 (--fp_interp 탄성 재질에만); 판 nofpi = F 전달 반대 (탄성 장면은 G2P 누적, 소성 장면은 가우스점 보간 강제)
    ELAS = scene in ("L000", "pillow2sofa")
    c = subprocess.check_output([sys.executable, f"{HERE}/fpa_cmd.py", scene, "def" if base == "nofpi" else base, rep, out], text=True).strip()
    if base == "nofpi":
        if ELAS:
            assert c.count(" --fp_interp ") == 1; c = c.replace(" --fp_interp ", " ")
    elif " --fp_interp " in c:
        c = c.replace(" --K ", " --fpi_el 1 --K ", 1)
else:
    c = subprocess.check_output([sys.executable, f"{HERE}/fpa_cmd.py", scene, base, rep, out], text=True).strip()
c = " " + c + " "
for m in mods:
    if re.match(r"cg\d+$", m):
        assert re.search(r" --cg \d+ ", c); c = re.sub(r" --cg \d+ ", f" --cg {m[2:]} ", c)
    elif m.startswith("tp"):
        assert re.search(r" --tolP \S+ ", c); c = re.sub(r" --tolP \S+ ", f" --tolP {m[2:]} ", c)
    elif m == "nofpi":
        assert c.count(" --fp_interp ") == 1; c = c.replace(" --fp_interp ", " ")
    elif m == "pfx":
        if " --fp_interp " not in c: c = c.replace(" --K ", " --fp_interp --K ", 1)
    elif m == "ssw":
        if "ssfix_wrap.py" not in c: c = c.replace("python -u exe/", "python -u " + SSW + "exe/", 1)
    elif m == "mlow":
        if " --mlow " not in c: c = c.replace(" --cell ", " --mlow 1e-3 --cell ", 1)
    elif m == "nomlow":
        c = re.sub(r" --mlow \S+ ", " ", c)
    elif re.match(r"uel\d+$", m):
        assert re.search(r" --ue_loc \d+ ", c); c = re.sub(r" --ue_loc \d+ ", f" --ue_loc {m[3:]} ", c)
    elif re.match(r"wc\d$", m):
        assert re.search(r" --wrx_col \d+ ", c); c = re.sub(r" --wrx_col \d+ ", f" --wrx_col {m[2:]} ", c)
    elif m == "fel":                                   # 절 CM191: --fp_interp 는 탄성 재질에만 (실행기 --fpi_el 1)
        if " --fpi_el " not in c: c = c.replace(" --K ", " --fpi_el 1 --K ", 1)
    elif re.match(r"wrx\d$", m):
        assert re.search(r" --wrx \d+ ", c); c = re.sub(r" --wrx \d+ ", f" --wrx {m[3:]} ", c)
    elif m == "dump":
        if " --dump " not in c: c = c.replace(" --K ", " --dump --dump_F --dump_part --K ", 1)
    else:
        raise SystemExit(f"모르는 토막 {m}")
print(" ".join(c.split()))
