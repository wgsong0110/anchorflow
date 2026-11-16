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
elif os.environ.get("FFX_ND", "0") in ("1", "2"):       # 2 = 절 CM192 새 기본 판 (아래에서 --wrx_el 1 을 더함)              # 절 CM191 새 기본 판 = CM190 기본 판 + --fpi_el 1 (--fp_interp 탄성 재질에만); 판 nofpi = F 전달 반대 (탄성 장면은 G2P 누적, 소성 장면은 가우스점 보간 강제)
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
if os.environ.get("FFX_ND", "0") == "2" and " --wrx_col " in c and " --wrx_el " not in c:   # 절 CM192 새 기본 판 = CM191 + --wrx_el 1 (탄성 재질은 X3 를 무너진 프레임에서만)
    c = c.replace(" --wrx_col ", " --wrx_el 1 --wrx_col ", 1)
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
    elif m == "wel":                                    # 절 CM192: 탄성 재질은 X3 를 무너진 프레임에서만 (--wrx_el 1)
        if " --wrx_el " not in c: c = c.replace(" --wrx_col ", " --wrx_el 1 --wrx_col ", 1)
    elif m in ("sstol", "sstolp"):                     # 절 CM192: 뉴턴 (·PCG) 허용치 x 1/K (exe/ssfix_wrap.py --ssf_tol 1 [--ssf_tolp 1]; K=1 이면 그대로)
        assert "--ssf_bec " in c; c = c.replace("--ssf_bec ", ("--ssf_tol 1 --ssf_tolp 1 " if m == "sstolp" else "--ssf_tol 1 ") + "--ssf_bec ", 1)
    elif m == "bec2":                                   # 절 CM192: 국소 후퇴 오일러 강제는 다시 풀기에서만
        assert "--ssf_bec 1 " in c; c = c.replace("--ssf_bec 1 ", "--ssf_bec 2 ", 1)
    elif m == "ssr":                                    # 절 CM192: ssfix 다시 풀기 뒤 X3 상태 되돌림
        assert "--ssf_bec 1 " in c; c = c.replace("--ssf_bec 1 ", "--ssf_bec 1 --ssf_wrxr 1 ", 1)
    elif re.match(r"skc[0-9.e-]+$", m):                 # 절 CM192: X1 고착 상한 문턱 --wrx_skc
        c = c.replace(" --wrx_col ", f" --wrx_skc {m[3:]} --wrx_col ", 1)
    elif m in ("alog", "best"):                         # 절 CM193: ssfix 시도별 기록 (결과 불변) / 끝까지 무너진 프레임은 가장 덜 나쁜 시도를 받음
        assert "--ssf_bec " in c; c = c.replace("--ssf_bec ", f"--ssf_{m} 1 --ssf_bec ", 1)
    elif re.match(r"sdg\d+-\d+$", m):                 # 절 CM193 진단 (결과 불변): exe/ssdiag_wrap.py 반복별 |g|²·F² 기록 (프레임 a..b, 1 부터)
        a_, b_ = m[3:].split("-"); assert "exe/startpt_wrap.py" in c
        c = c.replace("exe/startpt_wrap.py", f"exe/ssdiag_wrap.py --sd_f0 {a_} --sd_f1 {b_} exe/startpt_wrap.py", 1)
    elif re.match(r"xg[0-9.e-]+$", m):                  # 절 CM193: --xit_from 문 문턱 (ssfix --ssf_xgT)
        assert "--ssf_bec " in c; c = c.replace("--ssf_bec ", f"--ssf_xgT {m[2:]} --ssf_bec ", 1)
    elif m == "gbe":                                    # 절 CM193 진단: 후퇴 오일러 적분기 (fpa_cmd 판 be 와 같은 바꿈) 를 ssN 위에
        assert " --ga hht " in c and " --ue be " in c; c = c.replace(" --ga hht ", " --ga be ").replace(" --ue be ", " --ue beall "); c = re.sub(r" --ue_loc \d+ ", " ", c)
    elif re.match(r"rho\d+$", m):                      # 절 CM193 진단: HHT ρ∞ = 숫자/10 (rho5 = 0.5)
        assert re.search(r" --rho \S+ ", c); c = re.sub(r" --rho \S+ ", f" --rho {int(m[3:]) / 10:g} ", c)
    elif re.match(r"bs\d$", m):                        # 절 CM193: --be_sub N (K > 1 걸음 후퇴 오일러)
        assert " --K " in c; c = c.replace(" --K ", f" --be_sub {m[2:]} --K ", 1)
    elif m == "nofel":                                  # 절 CM193 진단: --fpi_el 1 뺌 (반환 사상 재질도 가우스 점 F 보간 교체)
        assert " --fpi_el 1 " in c; c = c.replace(" --fpi_el 1 ", " ", 1)
    elif m == "rmpos":                                  # 절 CM193: 모래 반환 사상 특이값 부호를 PhysGaussian 공식대로 (env AF_AO_RMPOS=1, 절 CM187 R)
        c = " AF_AO_RMPOS=1" + c
    elif m == "dump":
        if " --dump " not in c: c = c.replace(" --K ", " --dump --dump_F --dump_part --K ", 1)
    else:
        raise SystemExit(f"모르는 토막 {m}")
print(" ".join(c.split()))
