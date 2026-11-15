#!/usr/bin/env python3
"""절 CM183 ablation 명령 생성. 사용: abl_gen.py <장면> <판> <반복 번호> <출력 디렉토리>  -> 표준출력에 명령줄.
장면 = ficus bread wolf pillow2sofa vasedeck plane. 기본 명령 = exe/pr_ours_tmpl.json (R2 cm157/cm139 원본) + exe/pr2_grid.py 의 새 장면 유도 (그대로 복사).
판: def | q1 (가우스 1x1x1, --q1pt) | q27 (3x3x3) | q64 (4x4x4) | rho0 (후퇴 오일러) | rho1 (중점) | c075 c15 c20 (칸 크기 배수) | ssN (서브스텝 N, 나중)
반복 1 만 덤프 (--dump --dump_F --dump_part) + 풀이기 정보 (q27 0 감싸개, 결과 불변). 기본 판 불변, 바뀌는 것은 이 판의 플래그뿐."""
import json, os, re, sys
scene, var, rep, out = sys.argv[1:5]
rep = int(rep)
if var == "defb":   # CM183: 기본 판 (def) 과 같은 명령 -- 다른 기계에서 B·C 표의 기준 짝을 따로 재려고 이름만 다름
    var = "def"
HERE = os.path.dirname(os.path.abspath(__file__))
if var == "ak":
    # CM183 A1 (입자 적분점 판 = 절 AK, 실행기 exe/fem_ex_sim.py). 단일 차이 ablation 이 아님: 옛 명시적 symplectic + 고정 Kuhn 사면체 + P2G/G2P 는 유지, 셀 에너지만 입자 적분점 (Σ w_p ψ(A_T F_p)).
    # 명령 = R2 원본 cmd.txt 그대로 (ficus: lbours/fem_ak_ficus/ficus_cm05_r1 = 절 AK 최종 기본 판 VAR['cm05'] 전개, 커밋 46bd2fb;
    #        bread: lbours/fem_ag_bread/ag_bread_n160_pfm_r1 = 절 AG.8 --ak_pq_plast frame + --ah_mlow 0.1, 커밋 437e4a6), 데이터 경로만 vast_abl.sh 의 것으로, --out 바꿈.
    # 정답·오차 정의: 같은 데이터 .pt (d["gt"]), 같은 오차 (프레임별 입자 평균 |x-x_gt|/EXT 의 평균, fem_ex_sim 머리말 "fem_rt_sim 과 같은 지표"). 장면은 ficus·bread 만 (나머지: 빈 출력 + 종료코드 3 = 입자 구속 BC·소성·접촉 미지원).
    PYB = "/home/dkta/.conda/envs/af/bin/python"
    AKC = dict(
        ficus="%s -u exe/fem_ex_sim.py --out @OUT --data /dev/shm/dkta/anerr_data/F_pg_v0.pt --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --N 640 "
              "--mode lag --opt --mass tet --rstat --cfuse --cfuse_mode 2 --sp vcap --sp_vcap 1e9 --lim_sr 1e30 --ak_pq --ak_pq4 --ak_fstep --ah_fast --ak_prep --ak_fuse --sp_vcap 0.5 --ak_capmu @DUMP --ak_en 0 --ai_rigid 8" % PYB,
        bread="%s -u exe/fem_ex_sim.py --out @OUT --data /dev/shm/dkta/ag_data/bread.pt --tag bread --frames 300 --node_h 0.018856 --floor none --pbc --pg_vdamp --plast --mode lag --opt --mass tet --rstat "
              "--lim_sr 5 --cfuse --cfuse_mode 2 --sp vcap --sp_vcap 0.5 @DUMP --N 160 --lim_sr 1e30 --sp_vcap 1e9 --ak_pq --ak_pq_plast frame --ah_mlow 0.1" % PYB)
    if scene not in AKC:
        sys.exit(3)
    # 반복 1 만 덤프 (roll_fem.pt: 위치·F·v, F 는 float32; abl_an.py 가 읽음), 반복 2·3 은 덤프 없음 (시간 판)
    print(" ".join(AKC[scene].replace("@OUT", out).replace("@DUMP", "--dump_f32" if int(rep) == 1 else "--no_dump").split()))
    sys.exit(0)
T = json.load(open(f"{HERE}/pr_ours_tmpl.json"))
NW = "/home/dkta/work/newsc"
NS = dict(vasedeck=dict(fr=75, base="ficus", drop=["--region_density"]), plane=dict(fr=400, base="bread", drop=[]),
          pillow2sofa=dict(fr=300, base="ficus", drop=["--region_density"]))
if scene in NS:
    s = NS[scene]; c = T[s["base"]]["cmd"]
    c = re.sub(r"AF_SP_FR=1,\d+", f"AF_SP_FR=1,{s['fr']}", c)
    c = re.sub(r"--rr_shape \S+", f"--rr_shape {scene}", c)
    c = re.sub(r"--rr_traj \S+", f"--rr_traj {NW}/gt_{scene}", c)
    c = re.sub(r"--rr_cfg \S+", f"--rr_cfg {NW}/gt_{scene}/config.json", c)
    c = re.sub(r"--data \S+", f"--data {NW}/{scene}.pt", c)
    c = re.sub(r"--frames \d+", f"--frames {s['fr']}", c)
    c = re.sub(r"--tag \S+", f"--tag {scene}", c)
    for f in s["drop"]:
        c = c.replace(" " + f, "")
elif scene in ("toast", "sand_castle"):                 # GF 장면: BA_1 명령 + exe/gfs_cmd.py 규칙 (exe/abl_tmpl_gf.json), 경로 = /root/gfs (exe/abl_gf_fetch.sh)
    import subprocess, tempfile
    G = json.load(open(f"{HERE}/abl_tmpl_gf.json")); S = G["scenes"][scene]; GW = "/root/gfs"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f_:
        f_.write(G["BA_1"])
    env = dict(os.environ, GFS_PREFIX=S["prefix"])
    c = subprocess.run([sys.executable, f"{HERE}/gfs_cmd.py", f_.name, f"{GW}/ag/{scene}.pt", str(S["frames"]), scene, out, str(S["floor"]), f"{GW}/ag/gt_{scene}",
                        f"{GW}/hf/config/{S['cfg']}", f"{GW}/hf/model/{scene}", f"{GW}/GaussianFluent"], capture_output=True, text=True, env=env, check=True).stdout
    os.unlink(f_.name)
else:
    c = T[scene]["cmd"]
c = " ".join(c.split())
c = c.replace(" --dump_part", "").replace(" --dump_F", "").replace(" --dump ", " ")
GF = scene in ("toast", "sand_castle")
SP = "python -u " if GF else "exe/startpt_wrap.py --sp 1 "           # GF 명령에는 startpt 감싸개가 없음 -> 사슬 맨 앞 (python -u 바로 뒤)
assert (GF or c.count(SP) == 1) and c.count(" exe/fem_an_rt_sim.py ") == 1 and c.count(" --K 1 ") == 1
def wrap(c, n, src_cls):   # 가장 안쪽 (실행기 앞, 글자) + 가장 바깥 (startpt 뒤, 클래스) 감싸개 쌍
    return (c.replace(" exe/fem_an_rt_sim.py ", f" {src_cls[0]} exe/fem_an_rt_sim.py ").replace(SP, SP + src_cls[1]))
if var == "q1":
    c = wrap(c, 0, ("exe/q27_wrap.py --q1pt 1 --q27dg 1 --q27ph src", "exe/q27_wrap.py --q1pt 1 --q27dg 1 --q27ph cls "))
elif var in ("q27", "q64"):
    n = 3 if var == "q27" else 4
    c = wrap(c, 0, (f"exe/q27b_wrap.py --q27 1 --q27n {n} --q27dg 1 --q27ph src", f"exe/q27b_wrap.py --q27 1 --q27n {n} --q27dg 1 --q27ph cls "))
elif rep == 1:   # 기본 판 계열: 풀이기 정보 저장만 (결과 불변), 모래시계 분석용
    c = c.replace(SP, SP + "exe/q27_wrap.py --q27 0 --q27dg 1 ")
if var in ("rho0", "rho1"):
    # CM183 고침: HHT-α 꼴은 ρ∞=0 이 α=1 (분모 (1-α_f)=0 → ZeroDivisionError, 확인: R2 ablation/runs/ficus_rho0_r1/log.txt) 이라 ρ∞=0 은 Chung-Hulbert 일반화-α (--ga ga, 같은 실행기·증분 포텐셜 경로) 로.
    # ρ∞=1 은 HHT 그대로 (α=0, 뉴마크 평균 가속도 = 중점/사다리꼴).
    assert c.count(" --rho 0.8 ") == 1 and c.count(" --ga hht ") == 1
    c = c.replace(" --rho 0.8 ", f" --rho {var[-1]} ")
    if var == "rho0":
        c = c.replace(" --ga hht ", " --ga ga ")
m = re.match(r"c(\d+)$", var)
if m:
    mult = {"075": 0.75, "15": 1.5, "20": 2.0}[m.group(1)]
    base = [float(x) for x in re.findall(r"--cell (\S+)", c)][-1]
    c = re.sub(r" --cell \S+", "", c)
    c = c.replace(" --mg_levels ", f" --cell {base * mult:g} --mg_levels ", 1)
m = re.match(r"ss(\d+)$", var)
if m:
    c = c.replace(" --K 1 ", f" --K {m.group(1)} ", 1)
if rep == 1:
    c = c.replace(" --K ", " --dump --dump_F --dump_part --K ", 1) if "--dump" not in c else c
c = re.sub(r"--out \S+$", f"--out {out}", c)
print(c)
