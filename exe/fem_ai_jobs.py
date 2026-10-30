"""절 AI 스윕 작업 목록 (ROOT TAG ARGS 한 줄씩) -- exe/fem_ai_gpu.sh 일꾼이 읽는다. 사용: python exe/fem_ai_jobs.py <단계> [씬,...]
공통: 확정 경로 (절 AD/AF) + 클램프 끔 (--lim_sr 1e30 = 비율 상한 무한대: 판정 커널은 돌지만 작동 0 -- lim_n_tot 로 확인), 공통 h_s 6.25e-5 s.
판: A ∈ {a0 유클리드, a1 분리 평균, a2 응력 평균, a12 둘 다} x B ∈ {b0 A_T F, b1 navg (절 AB), b2 B-스플라인 기울기, b3 편차 분리} x 부피 상한 θ ∈ {v 0.5, n 끔}."""
import sys

W = "/home/dkta/work"
SC = {
    "L000": f"--data {W}/one/L000/L000.pt --tag L000 --floor vel --node_h 0.028284 --frames 80 --N 267",
    "ficus": f"--data {W}/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --N 640",
}
SC["wm"] = "--data /tmp/dkta/ag_data/wm.pt --tag wm --frames 100 --node_h 0.0094281 --floor vel --vfl_mode slip --floor_plane 1,1,0.0200000,0,0,1 --plast --N 480"   # 절 AG 수박 (GF NACC) 그대로
for t in ("0001", "0003", "0015", "0017", "0019"):
    SC[t] = f"--data /tmp/dkta/af_data/{t}.pt --tag T{t} --floor vel --node_h 0.028284 --frames 80 --N 267"
BASE = "--mode lag --opt --mass tet --rstat --cfuse --cfuse_mode 2 --no_dump --ai_stat 10 --ai_stat_frames 1,2,3,5 --fspread 10"
AA = {"a0": "", "a1": "--ai_p2g rs", "a2": "--ai_stress", "a12": "--ai_p2g rs --ai_stress"}
BB = {"b0": "", "b1": "--fupd navg --fupd_b 1", "b2": "--ai_g2p bs", "b3": "--ai_g2p dev"}
VV = {"v": "--sp vcap --sp_vcap 0.5", "n": "--sp none"}
CL = {"off": "--lim_sr 1e30", "free": "--lim_sr 1e30", "c5": "--lim_sr 5"}   # off = 커밋 f37da7d 판 (1e30 이어도 뒤집힘은 막힘 = 반전만 막는 판), free = 완전 끔 (SR_OFF 뒤)


def line(sc, ak, bk, vk, ck="free", r=1, kind=""):
    tag = f"{sc}_{ak}{bk}_{vk}_{ck}_" + ({"t": "t", "F": "F"}.get(kind, "r")) + str(r)
    root = {"ficus": "fem_ai_ficus", "wm": "fem_ai_wm"}.get(sc, "fem_ai_lego")
    ex = " --ai_stat 0 --ai_stat_frames 1,2,3,4,5,6,7,8,10,15,20,30,40,60,80,100" if sc == "wm" else ""   # 수박: 착지 (프레임 4~5) 앞뒤 매 프레임
    if kind:                                                                         # t = 단독 측정 (진단 없음, 덤프 없음), F = 최종 (f32 덤프, 진단 없음)
        base = BASE.split(" --no_dump")[0] + (" --no_dump" if kind == "t" else " --dump_f32")
        return f"{root} {tag} {SC[sc]} {base} {AA[ak]} {BB[bk]} {VV[vk]} {CL[ck]}".replace("  ", " ")
    return f"{root} {tag} {SC[sc]} {BASE} {AA[ak]} {BB[bk]} {VV[vk]} {CL[ck]}{ex}".replace("  ", " ")


st = sys.argv[1]
scs = sys.argv[2].split(",") if len(sys.argv) > 2 else ["L000", "0019", "ficus"]
out = []
if st == "s1":                                                                       # 1 차: 전체 A x B x θ, 클램프 끔, 1 회 + 확정 판 (클램프 5, a0b0 v) 기준
    for sc in scs:
        out.append(line(sc, "a0", "b0", "v", "c5"))
        for vk in ("v", "n"):
            for ak in AA:
                for bk in BB:
                    out.append(line(sc, ak, bk, vk))
elif st.startswith("t:") or st.startswith("F:"):                                    # t:<판,...>:<회 수> / F:<판,...>:1
    kind, combos, nr = st.split(":")
    for sc in scs:
        for cmb in combos.split(","):
            ab, vk = cmb.split("_")[:2]
            ck = cmb.split("_")[2] if cmb.count("_") >= 2 else "free"
            ak, bk = ab[:ab.index("b")], ab[ab.index("b"):]
            for r in range(1, int(nr) + 1):
                out.append(line(sc, ak, bk, vk, ck, r, kind))
elif st.startswith("rep:"):                                                          # rep:<판,...>:<회 수> -- 판 = a12b3_v 꼴
    _, combos, nr = st.split(":")
    for sc in scs:
        for cmb in combos.split(","):
            ab, vk = cmb.split("_")[:2]
            ck = cmb.split("_")[2] if cmb.count("_") >= 2 else "free"
            ak, bk = ab[:ab.index("b")], ab[ab.index("b"):]
            for r in range(1, int(nr) + 1):
                out.append(line(sc, ak, bk, vk, ck, r))
if st == "wm":                                                                       # 수박: A2 는 FCR 전용이라 제외 -- A0/A1 x B0~B3 x 클램프 끔/5, θ 0.5 (절 AG 판 그대로)
    for ck in ("c5", "free"):
        for ak in ("a0", "a1"):
            for bk in BB:
                out.append(line("wm", ak, bk, "v", ck))
if st.startswith("rb:"):                                                             # rb:<판,...>:<회 수> -- 절 AH.8 잔류 변형·몸 높이 (--ai_rigid 8) x 바닥 (옛 / 새 = 절 AH --vfl_w gtk --vfl_rate --vfl_npd 0.005)
    _, combos, nr = st.split(":")
    for sc in scs:
        for fk, fa in (("", ""), ("g", " --vfl_w gtk --vfl_rate --vfl_npd 0.005")):
            for cmb in combos.split(","):
                ab, vk = cmb.split("_")[:2]
                ck = cmb.split("_")[2] if cmb.count("_") >= 2 else "free"
                ak, bk = ab[:ab.index("b")], ab[ab.index("b"):]
                for r in range(1, int(nr) + 1):
                    ln = line(sc, ak, bk, vk, ck, r)
                    root, tag, rest = ln.split(" ", 2)
                    tag = tag.replace(f"_r{r}", f"_rb{fk}{r}")
                    out.append(f"{root} {tag} {rest}{fa} --ai_rigid 8")
print("\n".join(out))
