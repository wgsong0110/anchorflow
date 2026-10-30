"""절 AK 작업 목록 (ROOT TAG ARGS 한 줄씩) -- exe/fem_ak_sweep.sh 일꾼이 읽는다. 사용: python exe/fem_ak_jobs.py <단계> [씬,...]
기준 = 절 AI 클램프 없는 판 (A1 회전·신장 분리 평균 + A2 입자 응력 평균 + B1 매끈한 기울기 F 갱신, θ 0.5, 클램프 끔), 공통 h_s 6.25e-5 s (lego N267, ficus N640).
판 (원인 하나씩 바꿈): base / aff (프레임 단위 APIC) / b0 (B1 끔) / b0c5 (B1 끔 + 클램프 5) / a0 (A1·A2 끔) / a1 (A2 끔) / th1 (θ 1) / thn (상한 없음) / slip (바닥 미끄럼) / cfm (확정 설정)
바닥: 옛 (절 AF 속도 바닥) / g (절 AH 새 바닥 --vfl_w gtk --vfl_rate --vfl_npd 0.005)."""
import os
import sys

W = "/home/dkta/work"
SC = {
    "L000": f"--data {W}/one/L000/L000.pt --tag L000 --floor vel --node_h 0.028284 --frames 80 --N 267",
    "ficus": f"--data {W}/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp --N 640",
}
for t in ["%04d" % k for k in range(20)]:
    SC[t] = f"--data /tmp/dkta/af_data/{t}.pt --tag T{t} --floor vel --node_h 0.028284 --frames 80 --N 267"
NX = {"L000": 267, "ficus": 640}
PATH = "--mode lag --opt --mass tet --rstat --cfuse --cfuse_mode 2"
AI = "--ai_p2g rs --ai_stress --fupd navg --fupd_b 1 --sp vcap --sp_vcap 0.5 --lim_sr 1e30"
CFM = "--sp vcap --sp_vcap 0.5 --lim_sr 5"
GFL = "--vfl_w gtk --vfl_rate --vfl_npd 0.005"
DIAG = "--no_dump --ak_en 1 --ai_rigid 8"
VAR = {
    "base": AI,
    "aff": AI + " --affine",
    "b0": AI.replace("--fupd navg --fupd_b 1 ", ""),
    "b0c5": AI.replace("--fupd navg --fupd_b 1 ", "").replace("--lim_sr 1e30", "--lim_sr 5"),
    "a0": AI.replace("--ai_p2g rs --ai_stress ", ""),
    "a1": AI.replace("--ai_stress ", ""),
    "th1": AI.replace("--sp_vcap 0.5", "--sp_vcap 1.0"),
    "thn": AI.replace("--sp vcap --sp_vcap 0.5", "--sp none"),
    "slip": AI + " --vfl_mode slip",
    "cfm": CFM,
}


def line(sc, vk, fk="", r=1, extra="", kind="r", tagx=""):
    root = ("fem_ak_ficus" if sc == "ficus" else "fem_ak_lego") + os.environ.get("AK_SFX", "")
    if sc == "ficus" and (fk or vk == "slip"):
        return None
    tag = f"{sc}_{vk}{('_' + fk) if fk else ''}{tagx}_{kind}{r}"
    diag = {"r": DIAG, "t": "--no_dump", "F": "--dump_f32"}[kind]
    args = f"{SC[sc]} {PATH} {VAR[vk]} {GFL if fk == 'g' else ''} {diag} {extra}"
    return f"{root} {tag} " + " ".join(args.split())


st = sys.argv[1]
scs = sys.argv[2].split(",") if len(sys.argv) > 2 else ["L000", "0001", "0003", "0015", "0017", "0019", "ficus"]
out = []
if st == "s1":                                                                       # 1 차: 원인 분해 -- 판 x 바닥 x 씬, 1 회
    for sc in scs:
        for fk in ("", "g"):
            for vk in VAR:
                out.append(line(sc, vk, fk))
elif st.startswith("rs:"):                                                           # rs:<판,...>:<t0,...> -- 정답 프레임 t0 에서 시작 (회복만)
    _, vks, t0s = st.split(":")
    for sc in scs:
        for fk in ("", "g"):
            for vk in vks.split(","):
                for t0 in t0s.split(","):
                    out.append(line(sc, vk, fk, extra=f"--ak_t0 {t0}", tagx=f"_s{t0}"))
elif st.startswith("v:"):                                                            # v:<판,...>:<바닥 ,g>:<회 수>:<추가 인자 (@ 는 공백)>:<태그 접미>
    _, vks, fks, nr, ex, tx = (st.split(":") + ["", "", ""])[:6]
    for sc in scs:
        for fk in fks.split(","):
            for vk in vks.split(","):
                for r in range(1, int(nr) + 1):
                    out.append(line(sc, vk, fk, r, extra=ex.replace("@", " "), tagx=tx))
print("\n".join(q for q in out if q))
