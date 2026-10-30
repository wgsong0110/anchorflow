"""절 AH 작업 목록 (비측정 스윕): 한 설정으로 lego 홀드아웃 20 궤적 + L000 + ficus F_pg_v0 모두 기준 만족 + 서브스텝 로버스트.
줄 형식: <EX_ROOT> <태그> <fem_ex_sim 인자...>. AH_SET 으로 차수, AH_V 로 판 이름 (쉼표) 고르기, AH_N 으로 h_s 배율 (쉼표, 기본 1).
공통 설정 (절 AF) = 절 AD 확정 플래그 + h_s 6.25e-5 s (lego N 267, ficus N 640); 판은 그 위에 한 가지씩 바꾼다."""
import os

W = "/home/dkta/work"
D = os.environ.get("AF_DATA", f"{W}/lbpt/test")
REC = "--mode lag --opt --mass tet --rstat --lim_sr 5 --cfuse --cfuse_mode 2 --sp vcap --sp_vcap 0.5"
HS = 6.25e-5
LEGO = "--floor vel --node_h 0.028284 --frames 80 --no_dump --rbdiag"
FIC = (f"--data {W}/lbours/ficus_v0/data/F_pg_v0/F_pg_v0.pt --tag F_pg_v0 --frames 125 --node_h 0.056569 --floor none --cfg_bc --region_density --pg_vdamp "
       "--no_dump --dump_pred")
TR = [f"{i:04d}" for i in range(20)]


def rep(base, **kw):
    """REC 의 플래그를 바꾼 문자열 (값 None = 지움)."""
    t = base.split()
    o = []
    i = 0
    while i < len(t):
        k = t[i]
        if k.startswith("--") and k[2:] in kw:
            has_v = i + 1 < len(t) and not t[i + 1].startswith("--")
            i += 2 if has_v else 1
            continue
        o.append(k)
        i += 1
    for k, v in kw.items():
        if v is None:
            continue
        o.append(f"--{k}" + ("" if v is True else f" {v}"))
    return " ".join(o)


# 1 차: 원인 확인 -- 하나씩 끄거나 바꾸기 (판 이름 -> (플래그, ficus 도 돌릴지))
V1 = {
    "b0": (REC, False),                                  # 공통 설정 그대로 (강체 지표 기록용)
    "c10": (rep(REC, lim_sr=10), True),                  # 클램프 느슨하게
    "c20": (rep(REC, lim_sr=20), True),
    "keep": (REC + " --lim_v keep", True),               # 클램프가 속도를 줄이지 않음 (위치만)
    "full": (REC + " --vfl_mode full", False),           # 바닥: 접촉 입자 걸린 노드 속도 0 (정답 노드 sticky 꼴)
    "np0": (REC + " --vfl_np 0", False),                 # 바닥: 비관통 끔
    "slip": (REC + " --vfl_mode slip", False),           # 바닥: 미끄럼 (고착 없음, 민감도)
    "vc75": (rep(REC, sp_vcap=0.75), True),              # 셀 탄성 부피 상한
    "vc10": (rep(REC, sp_vcap=1.0), True),               #   상한 없음과 같음
    "h5": (REC + " --vp_h5 H5", False),                  # 정답 float32 위치로 V_p·m_p (버그 수정)
    "nb25": (REC + " --fupd navg --fupd_b 0.25", True),  # 절 AB 매끈한 F 갱신
}
# 2 차: 바닥 정의 정리 (fem_ah_cuda; 0bdde28 의 gtk·rate·gtkr* 태그는 노드 걸음 이중 적분 버그 판 -- 무효, 이름 바꿔 다시) -- 정답 sticky 격자 가중 (gtk) / 단위 시간당 고착 몫 (rate)
V2 = {
    "gk": (REC + " --vfl_w gtk", False),
    "rt": (REC + " --vfl_rate", False),
    "gkr": (REC + " --vfl_w gtk --vfl_rate", False),
    "gkr_np0": (REC + " --vfl_w gtk --vfl_rate --vfl_np 0", False),
    "gk_full": (REC + " --vfl_w gtk --vfl_mode full", False),
}
# 3 차: 비관통 깊이 (정답처럼 얕은 관통 허용) + 조합 ((d) F 갱신은 다른 에이전트 몫이라 뺌)
GKR = REC + " --vfl_w gtk --vfl_rate"
V3 = {
    "gkr_d05": (GKR + " --vfl_npd 0.01", False),
    "gkr_d025": (GKR + " --vfl_npd 0.005", False),
    "gkr_d10": (GKR + " --vfl_npd 0.02", False),
    "rt_d05": (REC + " --vfl_rate --vfl_npd 0.01", False),
    "gkr_vc75": (rep(GKR, sp_vcap=0.75), True),
    "gkr_d05_vc75": (rep(GKR, sp_vcap=0.75) + " --vfl_npd 0.01", False),
    "gkr_h5": (GKR + " --vp_h5 H5", False),
}
SETS = {"1": V1, "2": V2, "3": V3}
S = SETS[os.environ.get("AH_SET", "1")]
pick = [q for q in os.environ.get("AH_V", "").split(",") if q] or list(S)
NM = [float(q) for q in os.environ.get("AH_N", "1").split(",")]
R = int(os.environ.get("AH_R", "1"))
for r in range(1, R + 1):
    for nm in NM:
        hs = HS * nm
        NL, NF = round((1 / 60) / hs), round(0.04 / hs)
        sfx = "" if nm == 1 else f"_x{nm:g}"
        for v in pick:
            fl, fic = S[v]
            for t in TR + ["L000"]:
                dat = f"{W}/one/L000/L000.pt" if t == "L000" else f"{D}/{t}.pt"
                h5 = f"{W}/lbdata/test_low/L000/sim_0000000000.h5" if t == "L000" else f"{W}/lbdata/test/{t}/sim_0000000000.h5"
                print(f"fem_ah_lego ah_{v}{sfx}_{t}_r{r} --data {dat} --tag T{t} {LEGO} {fl.replace('H5', h5)} --N {NL}")
            if fic:
                print(f"fem_ah_ficus ah_{v}{sfx}_F_r{r} {FIC} {fl} --N {NF}")
